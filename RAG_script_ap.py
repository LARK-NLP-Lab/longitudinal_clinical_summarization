import torch
import argparse
import os
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain

from huggingface_hub import login
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pandas as pd

#for custom text loading
from langchain_core.document_loaders.base import BaseLoader
from langchain.schema import Document

torch.cuda.set_device(0)


class StringLoader(BaseLoader):
    def __init__(self, text: str):
        self.text = text

    def load(self):
        return [Document(page_content=self.text)]


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--inputdir', type=str, default='RAG', help='input directory')
    parser.add_argument('--inputdir', type=str, default='input_data', help='input directory')
    parser.add_argument('--outputdir', type=str, default='RAG_AP_output', help='output directory')
    parser.add_argument('--method', type=int, default=-1, help='AP generation method')
    parser.add_argument('--setting', type=str, default='gt', help='Experimental setting, gt or gen')
    args = parser.parse_args()
    return args


def model_setup():
    login()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=1000,
    )

    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return mistral_llm

def df2chron_str(df: pd.DataFrame):
    timestamps = df['REL_TIME'].to_list()
    text = df['TEXT'].to_list()

    chron_str = ''

    for x in zip(timestamps, text): 
        chron_str += "\t".join(map(str, x))
        chron_str += '\n'
    return chron_str

def get_retriever(chronology_str: str):
    # Load data
    loader = StringLoader(chronology_str)
    docs = loader.load()
    # Split text into chunks 
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)

    db = FAISS.from_documents(documents,
                            HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

    retriever = db.as_retriever()
    return retriever


def main():
    args = parse_args()
    input_folder = args.inputdir
    output_folder = args.outputdir
    method = args.method
    setting = args.setting
    
    mistral_llm = model_setup()

    # Create prompt template
    prompt_template = """
    ### [INST] Instruction: You are an experienced ICU clinician tasked with reviewing the following EHR data and generating concise Assessment and Plan sections of a clinical progress note. Use professional and medically appropriate language to provide a summary of the patient’s current status and the recommended course of action.
    {context}

    ### QUESTION:
    {question} [/INST]
    """

    # Create prompt from prompt template 
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Create llm chain
    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

    for filename in tqdm(os.listdir(input_folder)):
        df = pd.read_csv(f'{input_folder}/{filename}')
        day_groups = {day: group for day, group in df.groupby('DAY')}

        days = []
        gen_pns = []

        admission_id = filename.split('.')[0].split('_')[1]
        previous_pn = ''

        first_day = None
        for day, day_df in day_groups.items():
            if len(day_df[day_df['IS_NOTE'] == 1]) != 0:
                first_day = day
                break

        for day, day_df in day_groups.items():
            # skip days without progress notes
            if len(day_df[day_df['IS_NOTE'] == 1]) != 0:
                # grab all data that is not a progress note and turn it into a chron str
                ehr_str = df2chron_str(day_df.loc[day_df['IS_NOTE'] == 0])
                ehr_str += previous_pn
                
                # don't generate note for first day
                if day != first_day:
                    # get generated progress note from llm
                    retriever = get_retriever(ehr_str)
                    rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                        | llm_chain
                    )

                    assessment = rag_chain.invoke("Given the patient EHR data, write the Assessment section of a clinical progress note. The Assessment should include a brief description of both passive and active diagnoses. Clearly state why the patient is admitted to the hospital and describe the active problem for the day, along with any relevant comorbidities the patient has.")
                    plan = rag_chain.invoke("Given the patient EHR data, write the Plan section of a clinical progress note. The Plan should be organized into multiple subsections, each corresponding to a specific medical problem. Provide a detailed treatment plan for each problem, outlining proposed or ongoing interventions, medications, and care strategies.")
                    
                    text = '#1 Assessment:\n' + assessment['text'] + '\n'
                    text += '#2 Plan:\n' + plan['text'] + '\n'

                    # add to list for dataframe later
                    days.append(day)
                    gen_pns.append(text)
                

                if setting == 'gt':
                    # get ground truth PN for current day
                    next_prev = day_df[day_df['IS_NOTE'] == 1].iloc[-1]['TEXT']
                elif setting == 'gen':
                    if day == first_day:
                        # get gt for first day even if setting is gen
                        next_prev = day_df[day_df['IS_NOTE'] == 1].iloc[-1]['TEXT']
                    else:
                        next_prev = text
                
                # depending on method, add/append ground truth note
                if method == 1:
                    previous_pn = next_prev
                elif method == 2:
                    previous_pn += next_prev + '\n'
                
        output_df = pd.DataFrame(zip(days, gen_pns), columns=['DAY', 'TEXT'])
        output_df.to_csv(f'{output_folder}/genpns_{admission_id}.csv', index=False)
    return

if __name__ == '__main__':
    main()

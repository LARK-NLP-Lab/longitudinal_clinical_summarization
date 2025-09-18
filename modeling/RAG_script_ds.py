import torch
import argparse
import os
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain

from huggingface_hub import login
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pandas as pd

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
    parser.add_argument('--inputdir', type=str, default='input_data', help='input directory')
    parser.add_argument('--outputdir', type=str, default='RAG_output', help='output directory')
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
    
    mistral_llm = model_setup()

    # Create prompt template
    prompt_template = """
    ### [INST] Instruction: Answer the question based on the given context:

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
        chronology_str = df2chron_str(df)
        retriever = get_retriever(chronology_str)

        rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
            | llm_chain
        )
        diagnosis = rag_chain.invoke("What is the patient's main diagnosis?")
        bhc = rag_chain.invoke("Summarize the hospital course for this patient in a concise and accurate way.")
        dis_in = rag_chain.invoke("Given the input EHR data, generate discharge instructions for this patient.")

        complete_output = '#1 Diagnosis:\n' + diagnosis['text'] + '\n'
        complete_output += '#2 Brief Hospital Course:\n' + bhc['text'] + '\n'
        complete_output += '#3 Discharge Instructions:\n' + dis_in['text'] + '\n'

        admission_id = filename.split('.')[0].split('_')[1]
        with open(f'{output_folder}/rag_{admission_id}.txt', 'w') as text_file:
            text_file.write(complete_output)


if __name__ == '__main__':
    main()

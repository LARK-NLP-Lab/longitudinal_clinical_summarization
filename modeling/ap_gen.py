import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from huggingface_hub import login
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', type=str, default='data/AP/input', help='input directory')
    parser.add_argument('--method', type=int, default=-1, help='AP generation method')
    parser.add_argument('--setting', type=str, default='gt', help='Experimental setting, gt or gen')
    parser.add_argument('--model', type=str, help='model name, choose from mistral, qwen, deepseek, llama3, llama2')
    args = parser.parse_args()
    return args


def model_setup(model_selection: str):
    login()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    AVAILABLE_MODELS = {'mistral': 'mistralai/Mistral-7B-Instruct-v0.1',
                        'qwen': 'Qwen/Qwen2.5-VL-7B-Instruct',
                        'deepseek': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
                        'llama3': 'meta-llama/Llama-3.1-8B-Instruct',
                        'llama2': 'meta-llama/Llama-2-13b-chat-hf'}

    model_name = AVAILABLE_MODELS[model_selection]

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_8bit = AutoModelForCausalLM.from_pretrained(model_name,
                                                    quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print('Model and Tokenizer loading complete')
    return model_8bit, tokenizer, device


def df2chron_str(df: pd.DataFrame):    
    timestamps = df['REL_TIME'].to_list()
    text = df['TEXT'].to_list()

    chron_str = ''

    for x in zip(timestamps, text):
        chron_str += "\t".join(map(str, x))
        chron_str += '\n'
    return chron_str


def main():
    args = parse_args()
    input_folder = args.inputdir
    method = args.method
    setting = args.setting
    model_selection = args.model

    output_folder = f'data/AP/generated/DG/{model_selection}/{setting}/{method}'
    os.makedirs(output_folder, exist_ok=True) 

    model, tokenizer, device = model_setup(model_selection)
    
    instruction1 = """
    You are an experienced ICU clinician tasked with reviewing the following EHR data and generating concise Assessment and Plan sections of a clinical progress note. Use professional and medically appropriate language to provide a summary of the patient’s current status and the recommended course of action.    
    
    EHR Data:
    """

    instruction2 = """
    Assessment:
    The Assessment should include a brief description of both passive and active diagnoses. Clearly state why the patient is admitted to the hospital and describe the active problem for the day, along with any relevant comorbidities the patient has.
    Plan:
    The Plan should be organized into multiple subsections, each corresponding to a specific medical problem. Provide a detailed treatment plan for each problem, outlining proposed or ongoing interventions, medications, and care strategies.
    """

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
                    prompt = instruction1 + ehr_str + instruction2
                    inputs = tokenizer(prompt, return_tensors='pt').to(device)

                    outputs = model.generate(**inputs, max_new_tokens=1000)
                    text = tokenizer.batch_decode(outputs)[0]
                    text = text[len(prompt):]

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

                # get ground truth progress note - only for gt setting OR gen setting day 1
                # gt_pn = day_df[day_df['IS_NOTE'] == 1].iloc[-1]['TEXT']
                
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

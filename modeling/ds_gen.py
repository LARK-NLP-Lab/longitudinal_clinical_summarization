import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import argparse
from tqdm import tqdm
from huggingface_hub import login

torch.cuda.set_device(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', type=str, default='hand_selected/24h', help='input directory')
    parser.add_argument('--outputdir', type=str, default='mistral_summaries', help='output directory')
    args = parser.parse_args()
    return args


def model_setup():
    login()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'

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
    output_folder = args.outputdir

    model, tokenizer, device = model_setup()
    role = """
    Role: You are a clinician in the ICU responsible for generating patient discharge documentation.

    Hospital Data (last 24 hours before discharge):

    """

    instruction = """

    Part 1: Diagnosis
    As the clinician, provide a diagnosis that summarizes the patient's primary medical condition(s) identified during their hospital stay.

    ---

    Part 2: Hospital Course Summary
    Summarize the patient's hospital course over the past 24 hours, detailing key treatments administered, the patientâ€™s response, any significant events, and progress toward recovery.

    ---

    Part 3: Discharge Instructions
    Based on the diagnosis, hospital course, and specific patient data provided, write personalized discharge instructions. Include relevant medication guidelines, lifestyle and activity recommendations, follow-up appointments, and additional care advice tailored to the patient's condition and recent treatments. Ensure that instructions directly reflect any special considerations or precautions indicated by the patient's hospital course and condition to promote recovery and prevent readmission.

    """

    for filename in tqdm(os.listdir(input_folder)):
        chronology = pd.read_csv(f'{input_folder}/{filename}')
        chronology_str = df2chron_str(chronology)

        prompt = role + chronology_str + instruction
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        outputs = model.generate(**inputs, max_new_tokens=2000)
        text = tokenizer.batch_decode(outputs)[0]
        text = text[len(prompt):]

        admission_id = filename.split('.')[0].split('_')[1]
        with open(f'{output_folder}/gensummary_{admission_id}.txt', 'w') as text_file:
                text_file.write(text)


if __name__ == '__main__':
    main()

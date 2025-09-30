import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import argparse
from tqdm import tqdm
import gc
from huggingface_hub import login


EE_MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
EE_MAX_TOKENS = 2000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', type=str, default='/data/shiyue/PS/EE/ds/input/', help='input directory')
    parser.add_argument('--outputdir', type=str, default='/data/shiyue/PS/EE/ds/output/llama3/', help='output directory')
    args = parser.parse_args()
    return args

def model_setup():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        EE_MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(EE_MODEL_NAME)
    print("Model and tokenizer loaded")
    return model, tokenizer, device

def run_extraction(prompt, model, tokenizer, max_tokens=EE_MAX_TOKENS):
    messages = [
        {"role": "system", "content": "You are an experienced ICU physician reviewing the patient’s hospitalization in preparation for discharge. You will identify critical clinical events contributing to the patient's diagnosis, clinical decisions, and discharge plans."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    return generated_text

def extract_clinical_events(df, model, tokenizer):
    chronology_data = []
    for _, row in df.iterrows():
        chronology_data.append(f"{row['TIME']} | {row['TEXT']}")
    chronology_text = "\n".join(chronology_data)

    extraction_prompt = f"""DISCHARGE EVENT EXTRACTION TASK\n\nAnalyze the following data from the final 48 hours of the hospital stay and identify key clinical events that are most relevant for summarizing the course of treatment and informing discharge planning.\n\n{chronology_text}

Only include events that reflect:
1. Significant changes in symptoms or status (e.g., improvements, worsening, new findings)
2. Clinically important test results (especially abnormal values that lead to certain treatments)
3. Major treatments or interventions (e.g., medication changes, procedures, escalation/de-escalation)
4. Care team decisions that indicate readiness for discharge or change in care goals
5. Events linked to the final diagnosis or that inform follow-up care\n\nFormat:\n### Key Discharge Events ###\n- [Time]: [Category] [Description] (Reasoning)"""

    events_extracted = run_extraction(extraction_prompt, model, tokenizer, max_tokens=2000)
    return events_extracted

def df2chron_str(df: pd.DataFrame):
    timestamps = df['TIME'].to_list()
    text = df['TEXT'].to_list()

    chron_str = ''
    for x in zip(timestamps, text): 
        chron_str += "\t".join(map(str, x))
        chron_str += '\n'
    return chron_str

def generate_summary(model, tokenizer, device, chronology_str, extracted_events, question):
    try:
        full_context = f"### Patient Data:\n{chronology_str}\n\n### Extracted Clinical Events from Patient Data:\n{extracted_events}\n\n"
        prompt = f"### Instruction: {question}\n\n {full_context}### Response:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
        )

        text = tokenizer.batch_decode(outputs)[0]
        text = text[len(prompt):]

        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        return text
    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"ERROR: Unable to generate summary. {str(e)}"

def main():
    args = parse_args()
    input_folder = args.inputdir
    output_folder = args.outputdir
    os.makedirs(output_folder, exist_ok=True)

    model, tokenizer, device = model_setup()

    for filename in tqdm(os.listdir(input_folder)):
        if not filename.endswith('.csv'):
            continue

        file_path = os.path.join(input_folder, filename)
        try:
            df = pd.read_csv(file_path)
            if 'TIME' not in df.columns or 'TEXT' not in df.columns:
                print(f"Skipping file with missing columns: {filename}")
                continue

            df = df.sort_values(by=['TIME'])
            chronology_str = df2chron_str(df)
            print(f"Extracting events for {filename}...")
            extracted_events = extract_clinical_events(df, model, tokenizer)

            print(f"Generating discharge summary for {filename}...")
            diagnosis = generate_summary(
                model, tokenizer, device,
                chronology_str, extracted_events,
                "As the clinician, provide a diagnosis that summarizes the patient's primary medical condition(s) identified during their hospital stay."
            )

            hospital_course = generate_summary(
                model, tokenizer, device,
                chronology_str, extracted_events,
                "Summarize the patient's hospital course over the past 48 hours, detailing key treatments administered, the patient’s response, any significant events, and progress toward recovery."
            )

            discharge_instructions = generate_summary(
                model, tokenizer, device,
                chronology_str, extracted_events,
                "Based on the diagnosis, hospital course, and specific patient data provided, write personalized discharge instructions. Include relevant medication guidelines, lifestyle and activity recommendations, follow-up appointments, and additional care advice tailored to the patient's condition and recent treatments. Ensure that instructions directly reflect any special considerations or precautions indicated by the patient's hospital course and condition to promote recovery and prevent readmission."
            )

            complete_output = f"## 1. Diagnosis:\n{diagnosis}\n\n"
            complete_output += f"## 2. Hospital Course Summary:\n{hospital_course}\n\n"
            complete_output += f"## 3. Discharge Instructions;\n{discharge_instructions}\n\n"
            

            output_file_name = filename.replace('.csv', '.txt')
            output_file_path = os.path.join(output_folder, output_file_name)
            with open(output_file_path, "w") as text_file:
                text_file.write(complete_output)

            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

if __name__ == '__main__':
    main()


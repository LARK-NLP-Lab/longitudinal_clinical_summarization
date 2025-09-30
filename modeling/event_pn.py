import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import gc
from huggingface_hub import login

EE_MODEL_NAME = 'meta-llama/Llama-2-13b-chat-hf'
EE_MAX_TOKENS = 2000  

def model_setup_event_extraction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        EE_MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(EE_MODEL_NAME)
    print("Event extraction model and tokenizer loaded")
    return model, tokenizer


def run_extraction(prompt, model, tokenizer, max_tokens=EE_MAX_TOKENS):
    messages = [
         {"role": "system", "content": "You are an experienced clinician in the Intensive Care Unit (ICU). You will analyze the patient's clinical course using a structured Chain-of-Thought approach to identify critical clinical events contributing to the patient's diagnosis, clinical decisions, and recovery."},
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
   
    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()

    return generated_text


def extract_day_events(current_day, df):
    #relevant_data = df[df['DAY'].between(current_day - 3, current_day)]
    #relevant_data = df[df['DAY'].between(current_day, current_day + 3)]
    #relevant_data = df[df['DAY'] >= current_day]
    relevant_data = df[df['DAY'] == current_day]

    day_events = []
    for day, day_group in relevant_data.groupby('DAY'):
        day_events.append(f"=== DAY {int(day)} DATA ===")
        for _, row in day_group.iterrows():
            day_events.append(f"{row['TIME']} | {row['TEXT']}")
    day_events_text = "\n".join(day_events)
    
    extraction_prompt = f"""ICU DAILY EVENT EXTRACTION TASK

Analyze this structured ICU data by identifying critical clinical events, paying special attention to numerical values and their progression. Don't report every single lab value, instead, please only report numbers that show meaningful change or are clinically significant. For repeated values, only mention if they show change. Focus on:

{day_events_text}

Identify (with direct references to data points when possible):
1. Major symptoms/changes (new/worsening/improving) - Note specific values/changes
2. Critical test results (labs, imaging, etc.) - Highlight abnormal/normal values
3. Important treatments/interventions - Link to preceding data when evident
4. Significant care team decisions - Show supporting data if available
5. Major medical decisions/diagnosis/hypothesis - Reference relevant observations

Response Format:
### Day {int(current_day)} Key Events ###
- [Time]: [Event Category] [Description] (Explanation of identifying this event)
- Focus particularly on changes from previous days and new developments"""
    
    events_extracted = run_extraction(extraction_prompt, ee_model, ee_tokenizer, max_tokens=1000)
    return events_extracted


def model_setup_generation():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Generation device:", device)
    
    gen_model_name = EE_MODEL_NAME
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(gen_model_name, trust_remote_code=True)
    print("Generation model and tokenizer loaded")
    return model, tokenizer, device


def df2chron_str(df: pd.DataFrame):

    chron_str = ''
    for _, row in df.iterrows():
        chron_str += "\t".join(map(str, [row['REL_TIME'], row['TEXT']])) + "\n"
    return chron_str

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', type=str, default='/data/shiyue/PS/progress_note/input_data/input/', help='input directory')
    parser.add_argument('--outputdir', type=str, default='/data/shiyue/PS/EE/pn/llama2/', help='output directory')
    parser.add_argument('--method', type=int, default=-1, help='PN generation method')
    parser.add_argument('--setting', type=str, default='gt', help='Experimental setting, gt or gen')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    input_folder = args.inputdir
    setting = args.setting
    methods_to_run = [-1, 1, 2] 

    for method in methods_to_run:
        print(f"\n=== Running for method {method} under setting '{setting}' ===")
        
        if method == -1:
            method_folder = "method-1"
        else:
            method_folder = f"method{method}"
        output_folder = os.path.join(args.outputdir, f"{setting}_setting", method_folder)
        os.makedirs(output_folder, exist_ok=True)

        gen_model, gen_tokenizer, device = model_setup_generation()
        global ee_model, ee_tokenizer
        ee_model, ee_tokenizer = model_setup_event_extraction()

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

        for filename in tqdm(os.listdir(input_folder), desc=f"Files (method={method})"):
            file_path = os.path.join(input_folder, filename)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
                continue

            df = df.dropna(subset=["TEXT"])
            df = df.sort_values(by=["DAY", "TIME"])
            day_groups = {day: group for day, group in df.groupby('DAY')}
            days = []
            gen_pns = []

            admission_id = filename.split('.')[0].split('_')[1] if '_' in filename else filename.split('.')[0]
            previous_pn = ''

            first_day = None
            for day, group in day_groups.items():
                if len(group[group['IS_NOTE'] == 1]) != 0:
                    first_day = day
                    break

            for day, day_df in day_groups.items():
                if len(day_df[day_df['IS_NOTE'] == 1]) != 0:
                    events_extracted = extract_day_events(day, df)

                    ehr_str = df2chron_str(day_df[day_df['IS_NOTE'] == 0])
                    ehr_str += previous_pn

                    combined_input = (instruction1 + ehr_str +
                                      "\n\n=== Extracted Events ===\n" + events_extracted +
                                      "\n\n" + instruction2)

                    if day != first_day:
                        inputs = gen_tokenizer(combined_input, return_tensors='pt').to(device)
                        outputs = gen_model.generate(**inputs, max_new_tokens=1000)
                        generated_text = gen_tokenizer.batch_decode(outputs)[0]
                        gen_note = generated_text[len(combined_input):].strip()
                        days.append(day)
                        gen_pns.append(gen_note)

                        del outputs, inputs, generated_text
                        torch.cuda.empty_cache()
                        gc.collect()

                    if setting == 'gt':
                        next_prev = day_df[day_df['IS_NOTE'] == 1].iloc[-1]['TEXT']
                    elif setting == 'gen':
                        if day == first_day:
                            next_prev = day_df[day_df['IS_NOTE'] == 1].iloc[-1]['TEXT']
                        else:
                            next_prev = gen_note

                    if method == 1:
                        previous_pn = next_prev
                    elif method == 2:
                        previous_pn += next_prev + '\n'

            output_df = pd.DataFrame(list(zip(days, gen_pns)), columns=['DAY', 'TEXT'])
            output_filename = f"genpns_{admission_id}.csv"
            output_df.to_csv(os.path.join(output_folder, output_filename), index=False)

if __name__ == '__main__':
    main()

import re
import os
import numpy as np
import torch
from rouge import Rouge
from transformers import AutoTokenizer, AutoModel
from quickumls import QuickUMLS
import sys
import argparse


sys.setrecursionlimit(10000) 


rouge = Rouge(metrics=["rouge-l"])

# update with your own UMLS installation path
quickumls = QuickUMLS(
    "/path/to/QuickUMLS/",
    threshold=0.9,
    overlapping_criteria="score",
    similarity_name="cosine",
)


tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

SEMANTIC_TYPES = {
    "Diagnosis": ["T047", "T061","T023", "T170", "T201", "T048", "T074", "T121", "T031", "T060", "T056", "T034", "T029", "T046", "T191", "T190", "T184", "T033", "T037", "T041", "T109", "T190"],
    "Hospital Course": ["T121", "T170", "T033", "T109", "T058", "T047", "T031", "T061", "T023", "T046", "T201", "T074", "T184", "T116", "T123", "T059", "T060", "T195", "T041", "T029"],
    "Discharge Instructions": ["T121", "T033", "T170", "T184", "T109", "T047", "T061", "T023", "T058", "T074", "T031", "T201", "T046", "T041", "T195", "T040", "T116", "T037", "T039", "T123"],
}


def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

# Find the three sections in discharge summaries and ground truth files
def extract_sections(text, is_ground_truth=False):
    sections = {"Diagnosis": "", "Hospital Course": "", "Discharge Instructions": ""}

    if is_ground_truth:
        diagnosis_start = re.search(r"(Discharge Diagnosis:|DISCHARGED DIAGNOSES:|FINAL DIAGNOSIS:|DISCHARGE DIAGNOSIS:)", text)
        hospital_course_start = re.search(r"(Brief Hospital Course:|HOSPITAL COURSE:)", text)
        discharge_instructions_start = re.search(r"(Discharge Instructions:|DISCHARGE PLAN:|DISCHARGE INSTRUCTIONS/FOLLOWUP:|RECOMMENDED FOLLOWUP:|FOLLOWUP: )", text)
        
        diagnosis_end = re.search(r"Discharge Condition:|RECOMMENDED FOLLOWUP:|DISCHARGE INSTRUCTIONS/FOLLOWUP:|DISCHARGE MEDICATIONS:|MEDICATIONS ON DISCHARGE:", text)
        hospital_course_end = re.search(r"(Discharge Medications:|DISCHARGE STATUS:|Medications on Admission:|CONDITION ON DISCHARGE:|DISCHARGE DIAGNOSES:|DISCHARGE CONDITION:)", text)
        discharge_instructions_end = re.search(r"Followup Instructions:|RECOMMENDED FOLLOW-UP", text)
        if not discharge_instructions_end:
            discharge_instructions_end = re.search(r"\[\*\*First Name|\[\*\*Name", text)

        if diagnosis_start and diagnosis_end:
            sections["Diagnosis"] = text[diagnosis_start.end():diagnosis_end.start()].strip()
        
        if hospital_course_start and hospital_course_end:
            sections["Hospital Course"] = text[hospital_course_start.end():hospital_course_end.start()].strip()
        
        discharge_instructions_text = ""
        if discharge_instructions_start and discharge_instructions_end:
            discharge_instructions_text = text[discharge_instructions_start.end():discharge_instructions_end.start()].strip()
        
        discharge_medications_start = re.search(r"(Discharge Medications:|MEDICATIONS AT THE TIME OF DISCHARGE:|MEDICATIONS ON DISCHARGE:|DISCHARGE MEDICATIONS:)", text)
        discharge_disposition_end = re.search(r"Discharge Disposition:|DISCHARGE STATUS:|FOLLOW-UP:|FOLLOW-UP PLANS:", text)
        if not discharge_disposition_end:
            discharge_disposition_end = re.search(r"the patient's condition", text)
        
        if discharge_medications_start and discharge_disposition_end:
            discharge_instructions_text += "\n" + text[discharge_medications_start.end():discharge_disposition_end.start()].strip()

        sections["Discharge Instructions"] = discharge_instructions_text
    
    else:
        diag_pattern = re.compile(r"Part 1: Diagnosis|## 1. Diagnosis:")
        hosp_pattern = re.compile(r"Part 2: Hospital Course Summary|## 2. Hospital Course Summary:")
        discharge_pattern = re.compile(r"Part 3: Discharge Instructions|## 3. Discharge Instructions;")

        # Find Part 1: Diagnosis (first occurrence)
        diagnosis_match = diag_pattern.search(text)
        if diagnosis_match:
            # Find the first Part 2 marker occurring after Part 1
            hospital_course_match = hosp_pattern.search(text, pos=diagnosis_match.end())
            if hospital_course_match:
                sections["Diagnosis"] = text[diagnosis_match.end():hospital_course_match.start()].strip()
            else:
                sections["Diagnosis"] = text[diagnosis_match.end():].strip()
        else:
            hospital_course_match = None

        if hospital_course_match:
            # Find the first Part 3 marker after Part 2
            discharge_instructions_match = discharge_pattern.search(text, pos=hospital_course_match.end())
            if discharge_instructions_match:
                sections["Hospital Course"] = text[hospital_course_match.end():discharge_instructions_match.start()].strip()
            else:
                sections["Hospital Course"] = text[hospital_course_match.end():].strip()
        else:
            discharge_instructions_match = None

        if discharge_instructions_match:
            next_diagnosis_match = diag_pattern.search(text, pos=discharge_instructions_match.end())
            if next_diagnosis_match:
                sections["Discharge Instructions"] = text[discharge_instructions_match.end():next_diagnosis_match.start()].strip()
            else:
                sections["Discharge Instructions"] = text[discharge_instructions_match.end():].strip()
    
    return sections

def calculate_cui_fscore(generated_dir, ground_truth_dir):
    section_scores = {section: [] for section in SEMANTIC_TYPES}

    for gen_filename in os.listdir(generated_dir):
        if not gen_filename.endswith(".txt"):
            continue

        match = re.match(r"48h_all_abs_(\d+)\.txt$", gen_filename)
        if not match:
            continue
        base_id = match.group(1)
        gt_filename = f"gtsummary_{base_id}.txt"

        gen_text = read_file(os.path.join(generated_dir, gen_filename))
        gt_text = read_file(os.path.join(ground_truth_dir, gt_filename))

        gen_sections = extract_sections(gen_text, is_ground_truth=False)
        gt_sections = extract_sections(gt_text, is_ground_truth=True)

        for section, semantic_types in SEMANTIC_TYPES.items():
            if section in gen_sections and section in gt_sections:
                gen_cuis = extract_cuis_with_filter(gen_sections[section], semantic_types)
                gt_cuis = extract_cuis_with_filter(gt_sections[section], semantic_types)
                _, _, f1 = calculate_cui_metrics(gen_cuis, gt_cuis)
                section_scores[section].append(f1)

    results = {}
    for section, scores in section_scores.items():
        if scores:
            results[section] = (np.mean(scores), np.std(scores))
        else:
            results[section] = (0.0, 0.0)
    return results

def extract_cuis_with_filter(text, semantic_types):
    matches = quickumls.match(text, best_match=True, ignore_syntax=False)
    cuis = set()
    for match in matches:
        for entry in match:
            if any(semtype in entry["semtypes"] for semtype in semantic_types):
                cuis.add(entry["cui"])
    return cuis

def calculate_cui_metrics(pred_cuis, gold_cuis):
    intersection = pred_cuis.intersection(gold_cuis)
    precision = len(intersection) / len(pred_cuis) if pred_cuis else 0.0
    recall = len(intersection) / len(gold_cuis) if gold_cuis else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def calculate_rouge_l_for_all_files(generated_dir, ground_truth_dir):
    section_scores = {"Diagnosis": [], "Hospital Course": [], "Discharge Instructions": []}

    for gen_filename in os.listdir(generated_dir):
        if not gen_filename.endswith(".txt"):
            continue

        match = re.match(r"48h_all_abs_(\d+)\.txt$", gen_filename)
        if not match:
            continue
        base_id = match.group(1)
        gt_filename = f"gtsummary_{base_id}.txt"

        gen_text = read_file(os.path.join(generated_dir, gen_filename))
        gt_text = read_file(os.path.join(ground_truth_dir, gt_filename))

        gen_sections = extract_sections(gen_text, is_ground_truth=False)
        gt_sections = extract_sections(gt_text, is_ground_truth=True)

        for section in section_scores:
            if gen_sections[section] and gt_sections[section]:
                score = rouge.get_scores(gen_sections[section], gt_sections[section], avg=True)["rouge-l"]["f"]
                section_scores[section].append(score)

    results = {}
    for section, scores in section_scores.items():
        if scores:
            results[section] = (np.mean(scores), np.std(scores))
        else:
            results[section] = (0.0, 0.0)
    return results

def greedy_cos_idf(ref_embedding, ref_mask, ref_idf,
                   hyp_embedding, hyp_mask, hyp_idf,
                   all_layers=False):

    ref_embedding = ref_embedding / (ref_embedding.norm(dim=-1, keepdim=True) + 1e-8)
    hyp_embedding = hyp_embedding / (hyp_embedding.norm(dim=-1, keepdim=True) + 1e-8)
    
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    
    masks = torch.bmm(hyp_mask.unsqueeze(2).float(), ref_mask.unsqueeze(1).float())
    sim = sim * masks  

    word_precision = sim.max(dim=2)[0] 
    word_recall = sim.max(dim=1)[0]     

    hyp_idf = hyp_idf / (hyp_idf.sum(dim=1, keepdim=True) + 1e-8)
    ref_idf = ref_idf / (ref_idf.sum(dim=1, keepdim=True) + 1e-8)

    P = (word_precision * hyp_idf).sum(dim=1)  
    R = (word_recall * ref_idf).sum(dim=1)       
    F = 2 * P * R / (P + R + 1e-8)               

    return P, R, F

def evaluate_F1_greedy(ref_text, hyp_text, max_length=512, use_idf=True):
    ref_encoding = tokenizer(ref_text, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
    hyp_encoding = tokenizer(hyp_text, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
    
    with torch.no_grad():
        ref_outputs = model(**ref_encoding)
        hyp_outputs = model(**hyp_encoding)
    
    ref_embeddings = ref_outputs.last_hidden_state  
    hyp_embeddings = hyp_outputs.last_hidden_state  

    ref_mask = ref_encoding.attention_mask 
    hyp_mask = hyp_encoding.attention_mask 
    
    ref_idf = torch.ones_like(ref_mask, dtype=torch.float)
    hyp_idf = torch.ones_like(hyp_mask, dtype=torch.float)
    

    P, R, F = greedy_cos_idf(ref_embeddings, ref_mask, ref_idf,
                             hyp_embeddings, hyp_mask, hyp_idf,
                             all_layers=False)

    return F.item()

def calculate_average_bert_score(generated_dir, ground_truth_dir):
    section_scores = {"Diagnosis": [], "Hospital Course": [], "Discharge Instructions": []}

    for gen_filename in os.listdir(generated_dir):
        if not gen_filename.endswith(".txt"):
            continue

        match = re.match(r"48h_all_abs_(\d+)\.txt$", gen_filename)
        if not match:
            continue
        base_id = match.group(1)
        gt_filename = f"gtsummary_{base_id}.txt"

        gen_text = read_file(os.path.join(generated_dir, gen_filename))
        gt_text = read_file(os.path.join(ground_truth_dir, gt_filename))

        gen_sections = extract_sections(gen_text, is_ground_truth=False)
        gt_sections = extract_sections(gt_text, is_ground_truth=True)


        for section in section_scores:
            if gen_sections.get(section) and gt_sections.get(section):
                f1 = evaluate_F1_greedy(gt_sections[section], gen_sections[section])
                section_scores[section].append(f1)

    results = {}
    for section, scores in section_scores.items():
        if scores:
            results[section] = (np.mean(scores), np.std(scores))
        else:
            results[section] = (0.0, 0.0)
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate discharge summaries.")
    parser.add_argument("--gen_dir", type=str, required=True, help="Path to generated summaries.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to ground truth summaries.")
    args = parser.parse_args()

    avg_cui_scores = calculate_cui_fscore(args.gen_dir, args.gt_dir)
    avg_rouge_scores = calculate_rouge_l_for_all_files(args.gen_dir, args.gt_dir)
    avg_sapbert_scores = calculate_average_bert_score(args.gen_dir, args.gt_dir)

    print("\nEvaluation Results:")
    print("\nCUI F-scores:")
    for section, (mean_val, std_val) in avg_cui_scores.items():
        print(f"{section}: {mean_val * 100:.2f} ± {std_val * 100:.2f}")

    print("\nROUGE-L F-scores:")
    for section, (mean_val, std_val) in avg_rouge_scores.items():
        print(f"{section}: {mean_val * 100:.2f} ± {std_val * 100:.2f}")

    print("\nSapBERT F-scores:")
    for section, (mean_val, std_val) in avg_sapbert_scores.items():
        print(f"{section}: {mean_val * 100:.2f} ± {std_val * 100:.2f}")

if __name__ == "__main__":
    main()
import re
import os
import sys
import pandas as pd
import numpy as np
from rouge import Rouge
import torch
from transformers import AutoTokenizer, AutoModel
from quickumls import QuickUMLS
import sys
import argparse


sys.setrecursionlimit(10000)

rouge = Rouge(metrics=['rouge-l'])

# update with your own UMLS installation path
quickumls = QuickUMLS(
    "/path/to/QuickUMLS/",
    threshold=0.9,
    overlapping_criteria="score",
    similarity_name="cosine",
)

tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

SEMANTIC_TYPES = {"T170", "T033", "T074", "T047", "T201", "T046", "T184",
    "T031", "T029", "T023", "T041", "T048", "T061"}

def read_generated_text(file_path):
    df = pd.read_csv(file_path)
    return " ".join(df['TEXT'].astype(str).tolist())

def read_ground_truth_text(file_path):
    df = pd.read_csv(file_path)
    return " ".join(df['TEXT'].astype(str).tolist())

def calculate_rouge_l_f1(generated_dir, ground_truth_dir):
    f1_scores = []

    for gen_filename in os.listdir(generated_dir):
        if not gen_filename.startswith("genpns"):
            continue

        file_id = gen_filename.split("_")[1].replace(".csv", "")
        gt_filename = f"gt_{file_id}.csv"

        gen_file_path = os.path.join(generated_dir, gen_filename)
        gt_file_path = os.path.join(ground_truth_dir, gt_filename)

        gen_text = read_generated_text(gen_file_path)
        gt_text = read_ground_truth_text(gt_file_path)

        f1_score = rouge.get_scores(gen_text, gt_text, avg=True)['rouge-l']['f']
        f1_scores.append(f1_score)

    if f1_scores:
        return (np.mean(f1_scores), np.std(f1_scores))
    else:
        return (0.0, 0.0)

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

def evaluate_F1(ref_text, hyp_text, max_length=512, use_idf=True):
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

def calculate_average_bert_f1(generated_dir, ground_truth_dir):
    f1_scores = []

    for gen_filename in os.listdir(generated_dir):
        if not gen_filename.startswith("genpns"):
            continue

        file_id = gen_filename.split("_")[1].replace(".csv", "")
        gt_filename = f"gt_{file_id}.csv"  

        gen_file_path = os.path.join(generated_dir, gen_filename)
        gt_file_path = os.path.join(ground_truth_dir, gt_filename)

        gen_text = read_generated_text(gen_file_path)
        gt_text = read_ground_truth_text(gt_file_path)

        f1_score = evaluate_F1(gt_text, gen_text)
        f1_scores.append(f1_score)

    if f1_scores:
        return (np.mean(f1_scores), np.std(f1_scores))
    else:
        return (0.0, 0.0)

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

def calculate_cui_f1(generated_dir, ground_truth_dir):
    f1_scores = []

    for gen_filename in os.listdir(generated_dir):
        if not gen_filename.startswith("genpns"):
            continue

        file_id = gen_filename.split("_")[1].replace(".csv", "")
        gt_filename = f"gt_{file_id}.csv"

        gen_file_path = os.path.join(generated_dir, gen_filename)
        gt_file_path = os.path.join(ground_truth_dir, gt_filename)

        gen_text = read_generated_text(gen_file_path)
        gt_text = read_ground_truth_text(gt_file_path)
 
        gen_cuis = extract_cuis_with_filter(gen_text, SEMANTIC_TYPES)
        gt_cuis = extract_cuis_with_filter(gt_text, SEMANTIC_TYPES)

        _, _, f1 = calculate_cui_metrics(gen_cuis, gt_cuis)
        f1_scores.append(f1)

    if f1_scores:
        return (np.mean(f1_scores), np.std(f1_scores))
    else:
        return (0.0, 0.0)

def main():
    parser = argparse.ArgumentParser(description="Evaluate assessment and plan generation.")
    parser.add_argument("--gen_dir", type=str, required=True, help="Path to generated A&P.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to ground truth A&P.")
    args = parser.parse_args()

    # Evaluate each method and print scores
    for method in ['method-1', 'method1', 'method2']:
        method_dir = os.path.join(args.gen_dir, method)
        print(f"Evaluating -  {method}:")

        rouge_mean, rouge_std = calculate_rouge_l_f1(method_dir, args.gt_dir)
        bert_mean, bert_std   = calculate_average_bert_f1(method_dir, args.gt_dir)
        cui_mean, cui_std     = calculate_cui_f1(method_dir, args.gt_dir)

        print(f"  Average ROUGE-L F1 Score:  {rouge_mean * 100:.2f} ± {rouge_std * 100:.2f}")
        print(f"  Average SapBERT F1 Score:  {bert_mean * 100:.2f} ± {bert_std * 100:.2f}")
        print(f"  Average CUI-F1 Score:      {cui_mean * 100:.2f} ± {cui_std * 100:.2f}\n")

if __name__ == "__main__":
    main()
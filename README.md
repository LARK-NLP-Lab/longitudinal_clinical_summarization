# Clinical Summarization
## Overview
This repository contains code for the paper 'Large Language Models with Temporal Reasoning for Longitudinal Clinical Summarization and Prediction'

Link: https://arxiv.org/abs/2501.18724

## Data
We use the MIMIC-III dataset for this paper, which is available on PhysioNet: https://physionet.org/content/mimiciii/1.4/. To access the data, a credentialed PhysioNet account, CITI training and a data use agreement is required. For this reason the data cannot be included within this repository.
We also use EHRShot, which has similar requirements and can be found here: https://stanford.redivis.com/datasets/53gc-8rhx41kgt


## How to run
1. Run setup.sh to filter the MIMIC-III dataset and get the relevant chronologies for the discharge summarization and assessment and plan generation tasks. Modalities and time windows for discharge summarization can be selected by modifying the arguments (--modality and --window respectively) of get_chronologies_DS.py
2. To run generation, choose the bash script corresponding to the desired task (DS or AP) and setting (direct gen or RAG)

## Setting up UMLS
--insert description for UMLS installation--

## Tasks
- Discharge summarization: given a patient chronology, generate the three sections of a discharge summary (Diagnosis, Brief Hospital Course, Discharge Instructions)
- A&P generation: given a patient chronology and previous progress notes, generate the Assessment and Plan sections of the current day's progress note
- EHRSHot diagnosis prediction: given patient data, determine whether they will develop the given diagnosis within one year post-discharge.
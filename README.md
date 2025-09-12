# Clinical Summarization
## Overview
This repository contains code for the paper 'Zero-shot Large Language Models for Long Clinical Text Summarization with Temporal Reasoning'

Link: https://arxiv.org/abs/2501.18724

## Data
We use the MIMIC-III dataset for this paper, which is available on PhysioNet: https://physionet.org/content/mimiciii/1.4/. To access the data, a credentialed PhysioNet account, CITI training and a data use agreement is required. For this reason the data cannot be included within this repository.

## How to run
1. Run setup.sh to filter the MIMIC-III dataset and get the relevant chronologies for the discharge summarization and assessment and plan generation tasks. Modalities and time windows for discharge summarization can be selected by modifying the arguments (--modality and --window respectively) of get_chronologies_DS.py
2. To run generation, choose the bash script corresponding to the desired task (DS or AP) and setting (direct gen or RAG)


## Tasks
- Discharge summarization: given a patient chronology, generate the three sections of a discharge summary (Diagnosis, Brief Hospital Course, Discharge Instructions)
- A&P generation: given a patient chronology and previous progress notes, generate the Assessment and Plan sections of the current day's progress note
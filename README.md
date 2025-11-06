# Clinical Summarization
## Overview
This repository contains code for the paper 'Large Language Models with Temporal Reasoning for Longitudinal Clinical Summarization and Prediction'

Link: https://aclanthology.org/2025.findings-emnlp.1128/

## Data
We use the MIMIC-III dataset for this paper, which is available on PhysioNet: https://physionet.org/content/mimiciii/1.4/. To access the data, a credentialed PhysioNet account, CITI training and a data use agreement is required. For this reason the data cannot be included within this repository.
We also use EHRShot, which has similar requirements and can be found here: https://stanford.redivis.com/datasets/53gc-8rhx41kgt


## How to run
1. Run setup.sh to filter the MIMIC-III dataset and get the relevant chronologies for the discharge summarization and assessment and plan generation tasks. Modalities and time windows for discharge summarization can be selected by modifying the arguments (--modality and --window respectively) of get_chronologies_DS.py
2. To run generation, execute run.sh with the argument corresponding to the desired task ds | ap | rag_ds | rag_ap

## Setting up UMLS
As part of our evaluation metrics we calculate CUI f-score. This requires a working UMLS installation.


Download UMLS from the NIH website: https://www.nlm.nih.gov/research/umls/implementation_resources/metamorphosys/help.html


Then follow the instructions to install QuickUMLS https://github.com/Georgetown-IR-Lab/QuickUMLS

## Tasks
- Discharge summarization: given a patient chronology, generate the three sections of a discharge summary (Diagnosis, Brief Hospital Course, Discharge Instructions)
- A&P generation: given a patient chronology and previous progress notes, generate the Assessment and Plan sections of the current day's progress note
- EHRSHot diagnosis prediction: given patient data, determine whether they will develop the given diagnosis within one year post-discharge.

## Citation
If you use our work in your research, please cite the following paper:

```bibtex
@misc{kruse2025largelanguagemodelstemporal,
      title={Large Language Models with Temporal Reasoning for Longitudinal Clinical Summarization and Prediction}, 
      author={Maya Kruse and Shiyue Hu and Nicholas Derby and Yifu Wu and Samantha Stonbraker and Bingsheng Yao and Dakuo Wang and Elizabeth Goldberg and Yanjun Gao},
      year={2025},
      eprint={2501.18724},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.18724}, 
}
```

## Acknowledgement
This work is supported by National Library of Medicine R00 LM014308

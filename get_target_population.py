import pandas as pd
from datetime import datetime
from tqdm import tqdm
import csv
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=1000, help='number of admissions in sample')
    args = parser.parse_args()
    return args


def get_age(dob: str, cur: str):
    # calculates age at admission given date of birth
    format_str = r'%Y-%m-%d %H:%M:%S'
    dt_dob = datetime.strptime(dob, format_str)
    dt_cur = datetime.strptime(cur, format_str)
    difference = dt_cur - dt_dob
    age = int(difference.days / 365.25)
    return age


def filter_icu(target_path: str, sample_size: int) -> tuple[list[int], pd.DataFrame]:
    # get required files to filter patients by age, LOS, deceased status
    icu_df = pd.read_csv(f'{target_path}/ICUSTAYS.csv')
    admissions_df = pd.read_csv(f'{target_path}/ADMISSIONS.csv')        # needed to determine whether patient is deceased or not
    patient_df = pd.read_csv(f'{target_path}/PATIENTS.csv') 

    # filter ICU stays to satisfy criteria
    matched = icu_df.merge(patient_df[['SUBJECT_ID', 'DOB']], on='SUBJECT_ID', how='left')
    matched['age'] = matched.apply(lambda x: get_age(x['DOB'], x['INTIME']), axis=1)

    # filter by age and LOS
    filtered_icu = matched[matched["age"] >= 65]
    filtered_icu = filtered_icu[filtered_icu['LOS'] > 3]

    # filter out deceased patients
    alive_adm = admissions_df[admissions_df['DEATHTIME'].isna()]
    alive_icu = filtered_icu[filtered_icu['HADM_ID'].isin(alive_adm['HADM_ID'])]
    sampled_icu = alive_icu.sample(n=sample_size)

    # get list of HADMIDs from this filtered icu dataframe
    hadmids = sampled_icu['HADM_ID'].unique()
    return hadmids, sampled_icu


def main():
    args = parse_args()
    sample_size = args.sample_size

    target_path = 'data/MIMIC-III'

    hadmids, sampled_icu = filter_icu(target_path, sample_size)

    try:
        os.mkdir(f'{target_path}/filtered')
    except OSError:
        pass

    sampled_icu.to_csv(f'{target_path}/filtered/filtered_ICUSTAYS.csv', index=False)

    files = ['INPUTEVENTS_CV', 'INPUTEVENTS_MV', 'LABEVENTS', 'CHARTEVENTS', 'PRESCRIPTIONS', 'NOTEEVENTS']
    chunk_size = 10 ** 6  

    for file in files:
        print(f'Processing {file}...')
        new_df = pd.DataFrame()

        def write_csv(data):
            with open(f'{target_path}/filtered/filtered_{file}.csv', 'a') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(data)

        # Process the file in chunks
        for chunk in tqdm(pd.read_csv(f'{target_path}/{file}.csv', chunksize=chunk_size, low_memory=False)):
            target_file = chunk[chunk['HADM_ID'].isin(hadmids)]
            if not target_file.empty:
                new_df = pd.concat([new_df, target_file])
                write_csv(target_file)


if __name__ == '__main__':
    main()
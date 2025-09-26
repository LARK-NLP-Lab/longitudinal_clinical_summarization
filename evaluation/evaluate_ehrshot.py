import os
import pandas as pd
import re
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def create_diagnosis_dataframe(input_folder, conditions):
    # Initialize an empty dictionary to store diagnosis information for each patient
    patient_data = {}

    i = 0
    # Loop through each file in the directory
    for filename in os.listdir(input_folder):
        # Split the filename to get the condition and patient_id
        if filename.endswith('.txt'):
            filenames = filename.split('_')
            patient_id = filenames[-1]
            condition = filenames[0]
            if condition == 'acute':
                condition = condition + '_mi'
            condition = 'new_' + condition 
            patient_id = patient_id.replace('.txt', '')  # Remove the .txt extension
            
            diagnosis_value = 0
            with open(os.path.join(input_folder, filename), 'r') as file:
                # Skip the first empty line and read the second line (diagnosis)
                file.readline()  # Skip the first line (empty)
                file.readline()
                diagnosis_line = file.readline().strip()  # Read the second line
                i += 1
                print(i, diagnosis_line)
                # Check if the diagnosis line contains a valid diagnosis (either Positive or Negative)
                if diagnosis_line.startswith("Diagnosis:") or diagnosis_line.startswith("diagnosis:"):
                    diagnosis = diagnosis_line.split(":")[1].strip()  # Extract the diagnosis
                    if 'Positive' in diagnosis:
                        diagnosis_value = 1

            # Add the diagnosis to the dictionary, creating a new row if necessary
            if patient_id not in patient_data:
                patient_data[patient_id] = {condition: diagnosis_value}
            else:
                patient_data[patient_id][condition] = diagnosis_value

    # Convert the dictionary into a pandas DataFrame
    # Ensure all conditions are present as columns for every patient
    df = pd.DataFrame.from_dict(patient_data, orient='index', columns=conditions)

    # Fill missing columns with 0 (if any patient doesn't have a diagnosis for a certain condition)
    df = df.fillna(0).astype(int)

    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate EHRShot diagnosis prediction.")
    parser.add_argument("--gen_dir", type=str, required=True, help="Path to generated EHRShot output.")
    args = parser.parse_args()

    # Define conditions
    conditions = ['new_acute_mi', 'new_celiac', 'new_hyperlipidemia', 'new_hypertension', 'new_lupus', 'new_pancan']

    # Create the DataFrame
    df = create_diagnosis_dataframe(args.gen_dir, conditions)

    # insert path to gold data here - preprocessed EHRShot
    labels = pd.read_csv('processed_ehrshot.csv')
    labels.rename(columns={'new_acutemi' : 'new_acute_mi'}, inplace=True)

    filtered_labels = labels[labels['patient_id'].isin([int(x) for x in df.index.tolist()])]
    labels = filtered_labels.astype(int)
    df = df.reindex(labels['patient_id'].astype(str))

    for condition in conditions:
        y_true = labels[condition]
        y_pred = df[condition]
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"Accuracy for {condition}: {acc}")
        print(f"Precision for {condition}: {prec}")
        print(f"Recall for {condition}: {rec}")
        print(f"F1 Score for {condition}: {f1}")


if __name__ == '__main__':
    main()
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='both', help='modality - both, notes or tab')
    parser.add_argument('--window', type=int, default=24, help='temporal context window - 24 or 48')
    args = parser.parse_args()
    return args


def filter(labs: pd.DataFrame, charts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes duplicate instances that are present in both chart and lab events from charts.
    Measurements in chart events that have Warning == 0 are discarded. Warning == 1 or NaN are kept.
    Measurements in lab events with the 'abnormal' flag are kept, others discarded.

    Args:
        labs (pd.DataFrame): DataFrame containing lab events
        charts (pd.DataFrame): DataFrame containing chart events

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: modified lab and chart event DataFrames
    """
    duplicates = pd.merge(charts, labs, on=['CHARTTIME', 'ITEM_DESC'], how='inner')
    duplicates_row_IDs = duplicates['ROW_ID_x']

    charts = charts[~charts['ROW_ID'].isin(duplicates_row_IDs)]
    charts = charts[charts['WARNING'] != 0].reset_index(drop=True)
    labs = labs[labs['FLAG'] == 'abnormal']
    return labs, charts


def remove_duplicates_struc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate measurements (with the same value) in structured/tabular data 
    if they are present multiple times at the same timestamp or within one hour of each other. 

    Args:
        df (pd.DataFrame): DataFrame containing combined structured data

    Returns:
        pd.DataFrame: modified structured data DataFrame
    """
    meds = df[df['IS_MED'] == 1]
    new_df = df[df['IS_MED'] == 0]
    new_df['CHARTTIME'] = pd.to_datetime(new_df['CHARTTIME'])

    result = []
    for _, group in new_df.groupby(['VALUE', 'VALUEUOM', 'ITEM_DESC']):
        group = group.sort_values(by='CHARTTIME')
        keep_indices = []
        last_time = None

        for index, row in group.iterrows():
            if last_time is None or (row['CHARTTIME'] - last_time).total_seconds() > 3600:
                keep_indices.append(index)
            last_time = row['CHARTTIME']
        
        result.append(group.loc[keep_indices])
    
    # if no duplicates, return original input DataFrame
    if not result:
        return df

    # Combine all groups back into a single dataframe
    filtered_df = pd.concat(result).sort_index()
    filtered_df['CHARTTIME'] = filtered_df['CHARTTIME'].dt.strftime('%Y-%m-%d %H:%M:%S')
    filtered_df = pd.concat((filtered_df, meds))

    filtered_df = filtered_df.sort_values(by='CHARTTIME')
    return filtered_df


def temporal_order_struc(tab_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct temporally ordered structured data. Measurements are grouped by timestamp and
    converted to a narrative format. Measurements without timestamps are removed.

    Args:
        tab_df (pd.DataFrame): DataFrame containing combined structured data

    Returns:
        pd.DataFrame: DataFrame containing temporally ordered structured data in narrative format
    """
    # get unique timestamps, group by those. throw out empty values. convert to narrative format - different format if is_input == 1
    # truncate floats after two decimals

    timestamps = tab_df['CHARTTIME'].unique()
    text = []
    for time in timestamps:
        output_str = ''
        time_df = tab_df[tab_df['CHARTTIME'] == time]
        for i in time_df.index:
            value = time_df.loc[i, 'VALUE'] if not pd.isna(time_df.loc[i, 'VALUE']) else ''
            uom = time_df.loc[i, 'VALUEUOM'] if not pd.isna(time_df.loc[i, 'VALUEUOM']) else ''
            desc = time_df.loc[i, 'ITEM_DESC']
            # if value can be converted to float round to two decimal places
            if '.' in str(value):
                try:
                    value = f'{float(value):.2f}'
                except:
                    ValueError
            # different phrasing if input event or medication
            if time_df.loc[i, 'IS_INPUT'] == 0 and value != '':
                output_str += f'{desc} is {value} {uom}. '
            elif time_df.loc[i, 'IS_INPUT'] == 1:
                if value == '':    
                    output_str += f'{desc} is administered. '
                else:
                    output_str += f'{value} {uom} of {desc} is administered. '
            elif time_df.loc[i, 'IS_MED'] == 1:
                drug = time_df.loc[i, 'DRUG']
                value = time_df.loc[i, 'PROD_STRENGTH'] if not pd.isna(time_df.loc[i, 'PROD_STRENGTH']) else ''
                if value == '':    
                    output_str += f'{drug} is administered. '
                else:
                    output_str += f'{value} of {drug} is administered. '
        text.append(output_str)
    return pd.DataFrame(zip(timestamps, text), columns=['TIME', 'TEXT'])

def get_structured(hadmid: int, tab_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], lookup: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """
    For a given hadmid, gathers all structured data (lab, chart, input, medications) and combines it into
    one DataFrame. Applies filtering, converting to narrative format, and removal of duplicates.

    Args:
        hadmid (int): unique hospital admission ID
        tab_data (tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]): tuple containing DataFrames for lab, chart, input events + medications
        lookup (tuple[pd.DataFrame, pd.DataFrame]): tuple of DataFrames used as dictionaries to map item IDs to natural language descriptions

    Returns:
        pd.DataFrame: DataFrame of combined structured data, ordered by timestamp
    """
    # takes hospital admission and returns dataframe of chronologically ordered structured data
    lab_df, chart_df, input_df, meds_df = tab_data
    lab_items, chart_items = lookup
    
    # get structured data (lab, chart, input events for relevant hospital admission)
    patient_lab = lab_df[lab_df['HADM_ID'] == hadmid]
    patient_chart = chart_df[chart_df['HADM_ID'] == hadmid]
    patient_input = input_df[input_df['HADM_ID'] == hadmid]
    patient_input = input_df[input_df['HADM_ID'] == hadmid]
    patient_meds = meds_df[meds_df['HADM_ID'] == hadmid]

    # convert ITEMID to natural language description - column ITEM_DESC in dataframe
    patient_lab['ITEM_DESC'] = patient_lab['ITEMID'].apply(lambda x: lab_items.loc[lab_items['ITEMID'] == x]['LABEL'].values[0])
    patient_chart['ITEM_DESC'] = patient_chart['ITEMID'].apply(lambda x: chart_items.loc[chart_items['ITEMID'] == x]['LABEL'].values[0])
    patient_input['ITEM_DESC'] = patient_input['ITEMID'].apply(lambda x: chart_items.loc[chart_items['ITEMID'] == x]['LABEL'].values[0])

    patient_input = patient_input.rename(columns={'AMOUNT': 'VALUE', 'AMOUNTUOM': 'VALUEUOM'})
    patient_input['IS_INPUT'] = 1

    patient_meds['IS_MED'] = 1
    patient_meds = patient_meds.rename(columns={'STARTDATE': 'CHARTTIME'})

    patient_lab, patient_chart = filter(patient_lab, patient_chart)

    # consolidate all structured data into one df, order by timestamp
    patient_struc = pd.concat((patient_lab, patient_chart, patient_input, patient_meds))
    patient_struc['IS_INPUT'] = patient_struc['IS_INPUT'].apply(lambda x: 1 if x == 1 else 0)
    patient_struc['IS_MED'] = patient_struc['IS_MED'].apply(lambda x: 1 if x == 1 else 0)
    patient_struc = patient_struc.sort_values(by='CHARTTIME')
    patient_struc = patient_struc.reset_index(drop=True)

    # remove duplicate entries in structured data
    patient_struc = remove_duplicates_struc(patient_struc)

    struc_tl = temporal_order_struc(patient_struc)
    return struc_tl


def temporal_order_note(note_df: pd.DataFrame) -> pd.DataFrame:
    """
    Order note data by timestamp

    Args:
        note_df (pd.DataFrame): collected notes for one hospital admission

    Returns:
        pd.DataFrame: ordered notes 
    """
    timestamps = []
    text = []

    for i in note_df.index:
        ts = note_df.loc[i, 'CHARTTIME']
        cat = note_df.loc[i, 'CATEGORY']
        note_text = note_df.loc[i, 'TEXT']
        timestamps.append(ts)
        text.append(f'{cat} note: \n{note_text}')

    notes_tl = pd.DataFrame(zip(timestamps, text), columns=['TIME', 'TEXT'])
    notes_tl = notes_tl.sort_values(by='TIME')
    return notes_tl

def get_notes(hadmid: int, notes: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Extracts all notes for a given hadmid except discharge summaries. Removes duplicate notes at same timestamp.

    Args:
        hadmid (int): unique hospital admission ID
        notes (pd.DataFrame): DataFrame containing all notes in the dataset

    Returns:
        pd.DataFrame: DataFrame of all notes for the given hadmid
    """
    patient_notes = notes[notes['HADM_ID'] == hadmid]
    patient_notes = patient_notes[patient_notes['CATEGORY'] != 'Discharge summary']
    notes_tl = temporal_order_note(patient_notes)

    # remove notes at same timestamp
    notes_tl = notes_tl.drop_duplicates(subset=['TIME'], keep='last')

    # get ground truth discharge summary
    patient_notes = notes[notes['HADM_ID'] == int(hadmid)]
    discharge = patient_notes[patient_notes['CATEGORY'] == 'Discharge summary']
    if discharge.empty:
        print(f'{hadmid} has not discharge summary')                #very rare
    else:
        discharge_txt = patient_notes[patient_notes['CATEGORY'] == 'Discharge summary']['TEXT'].values[0]
    return notes_tl, discharge_txt


def get_rel_times(df):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    # convert absolute timestamps to relative ones (w.r.t first entry)
    format_str = r'%Y-%m-%d %H:%M:%S'
    rel_times = []
    for i in df.index:
        if i == 0:
            rel_times.append('First entry: ')
        else:
            previous_ts = str(df.iloc[i-1]['TIME'])
            current_ts = str(df.iloc[i]['TIME'])
            if current_ts != 'nan':
                previous_ts = datetime.strptime(previous_ts, format_str)
                current_ts = datetime.strptime(current_ts, format_str)
                difference = current_ts - previous_ts
                
                days = difference.days
                hours, remainder = divmod(difference.seconds, 3600)
                minutes = remainder // 60
                time_lst = []
                if days > 0:
                    time_lst.append(f"{days} day{'s' if days > 1 else ''}")
                if hours > 0:
                    time_lst.append(f"{hours} hour{'s' if hours > 1 else ''}")
                if minutes > 0:
                    time_lst.append(f"{minutes} minute{'s' if minutes > 1 else ''} later: ")
                rel_times.append(" ".join(time_lst))
            else:
                rel_times.append(None)
    df['REL_TIME'] = rel_times
    return df

def get_last_day(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # get data from last 24h before discharge/last entry
    format_str = r'%Y-%m-%d %H:%M:%S'

    for i in df.index[::-1]:
        if i == len(df)-1:
            last_entry = datetime.strptime(df.iloc[i]['TIME'], format_str)
        else:
            current_entry = datetime.strptime(df.iloc[i]['TIME'], format_str)
            diff_days = (last_entry - current_entry).days
            if window == 24:
                if diff_days != 0:
                    return df[i+1:].reset_index(drop=True)
            elif window == 48:
                if diff_days >= 2:
                    return df[i+1:].reset_index(drop=True)
    return df.reset_index(drop=True)


def format_result(df: pd.DataFrame) -> pd.DataFrame:
    # move relative time column to front
    col = df.pop('REL_TIME')
    df.insert(0, 'REL_TIME', col)
    return df


def main():
    pd.options.mode.chained_assignment = None  # Turns off the warning

    args = parse_args()
    modality = args.modality
    window = args.window

    target_path = 'data/target_population/filtered'

    icu_df = pd.read_csv(f'{target_path}/filtered_ICUSTAYS.csv')
    notes = pd.read_csv(f'{target_path}/filtered_NOTEEVENTS.csv')
    input_cv = pd.read_csv(f'{target_path}/filtered_INPUTEVENTS_CV.csv')
    input_mv = pd.read_csv(f'{target_path}/filtered_INPUTEVENTS_MV.csv')
    lab_df = pd.read_csv(f'{target_path}/filtered_LABEVENTS.csv')
    chart_df = pd.read_csv(f'{target_path}/filtered_CHARTEVENTS.csv')
    meds_df = pd.read_csv(f'{target_path}/filtered_PRESCRIPTIONS.csv')
    lab_items = pd.read_csv('data/MIMIC-III/D_LABITEMS.csv')
    chart_items = pd.read_csv('data/MIMIC-III/D_ITEMS.csv')

    admission_ids = icu_df['HADM_ID'].to_list()

    output_dir = 'discharge_sum/input'
    gt_dir = 'discharge_sum/gold'
    
    # for admission_id in tqdm(batch_ids):
    for admission_id in tqdm(admission_ids):
        admission_id = int(admission_id)
        dbsource = icu_df[icu_df['HADM_ID'] == int(admission_id)]['DBSOURCE'].values[0]
        if dbsource == 'carevue':
            input_df = input_cv
        else:
            input_df = input_mv

        tab_data = (lab_df, chart_df, input_df, meds_df)
        dictionaries = (lab_items, chart_items)

        structured_data = get_structured(admission_id, tab_data, dictionaries)
        note_data, gold_txt = get_notes(admission_id, notes)

        if modality == 'both':
            combined_tl = pd.concat((structured_data, note_data))
        elif modality == 'notes':
            combined_tl = note_data
        else:           # modality == tab
            combined_tl = structured_data
        
        combined_tl = combined_tl.sort_values(by='TIME').reset_index(drop=True)

        combined_tl_rel = get_rel_times(combined_tl)
        combined_tl_rel = combined_tl_rel.dropna(subset=['TIME'])

        last_day = get_last_day(combined_tl_rel, window)

        formatted_last_day = format_result(last_day)

        # write gold discharge summary to text file
        with open(f'{gt_dir}/gtsummary_{admission_id}.txt', 'w') as text_file:
            text_file.write(gold_txt)

        formatted_last_day.to_csv(f'{output_dir}/{window}_{modality}_{admission_id}.csv', index=False)


if __name__ == '__main__':
    main()
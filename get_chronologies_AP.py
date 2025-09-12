import pandas as pd
from datetime import datetime
from tqdm import tqdm


def filter(labs: pd.DataFrame, charts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # remove duplicates and keep only abnormal lab values
    # filtering by warning in charts - remove 0 values    

    duplicates = pd.merge(charts, labs, on=['CHARTTIME', 'ITEM_DESC'], how='inner')
    duplicates_row_IDs = duplicates['ROW_ID_x']

    charts = charts[~charts['ROW_ID'].isin(duplicates_row_IDs)]
    charts = charts[charts['WARNING'] != 0].reset_index(drop=True)
    labs = labs[labs['FLAG'] == 'abnormal']
    return labs, charts


def remove_duplicates_struc(df: pd.DataFrame) -> pd.DataFrame:
    meds = df[df['IS_MED'] == 1]
    df = df[df['IS_MED'] == 0]
    df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME'])

    # Group by VALUE, UNIT, and ITEM
    result = []
    for _, group in df.groupby(['VALUE', 'VALUEUOM', 'ITEM_DESC']):
        # Sort by TIME within each group (already sorted globally)
        group = group.sort_values(by='CHARTTIME')
        keep_indices = []
        last_time = None

        for index, row in group.iterrows():
            if last_time is None or (row['CHARTTIME'] - last_time).total_seconds() > 3600:
                keep_indices.append(index)
            last_time = row['CHARTTIME']
        
        result.append(group.loc[keep_indices])

    if not result:
        return df
    
    # Combine all groups back into a single dataframe
    filtered_df = pd.concat(result).sort_index()
    filtered_df['CHARTTIME'] = filtered_df['CHARTTIME'].dt.strftime('%Y-%m-%d %H:%M:%S')
    filtered_df = pd.concat((filtered_df, meds))

    filtered_df = filtered_df.sort_values(by='CHARTTIME')
    return filtered_df


def temporal_order_struc(tab_df: pd.DataFrame) -> pd.DataFrame:
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
            # different phrasing if input event
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
    # takes hospital admission and returns dataframe of chronologically ordered structured data
    lab_df, chart_df, input_df, meds_df = tab_data
    lab_items, chart_items = lookup
    
    # get structured data (lab. chart, input events for relevant hospital admission)
    patient_lab = lab_df[lab_df['HADM_ID'] == hadmid]
    patient_chart = chart_df[chart_df['HADM_ID'] == hadmid]
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

    # remove duplicate entries in structured data here
    patient_struc = remove_duplicates_struc(patient_struc)

    struc_tl = temporal_order_struc(patient_struc)
    return struc_tl


def get_prog_notes(hadmid: int, notes: pd.DataFrame) -> pd.DataFrame:
    patient_notes = notes[notes['HADM_ID'] == hadmid]
    patient_notes = patient_notes.sort_values(by='CHARTTIME')
    patient_notes = patient_notes.reset_index(drop=True)
    tl_prog = patient_notes[["CHARTTIME", 'TEXT']]
    tl_prog = tl_prog.rename(columns={'CHARTTIME': 'TIME'})
    tl_prog['IS_NOTE'] = 1
    return tl_prog


def temporal_order_note(note_df: pd.DataFrame) -> pd.DataFrame:
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


def get_ehr_notes(hadmid: int, notes: pd.DataFrame) -> pd.DataFrame:
    # only get non-PN notes
    patient_notes = notes[notes['HADM_ID'] == hadmid]
    ehr_notes = patient_notes[~patient_notes['DESCRIPTION'].str.contains(r'\bProgress Note\b', case=False)]
    ehr_notes = ehr_notes.dropna(subset='CHARTTIME')
    notes_tl = temporal_order_note(ehr_notes)

    # remove notes at same timestamp
    notes_tl = notes_tl.drop_duplicates(subset=['TIME'], keep='last')
    return notes_tl


def get_rel_times(df):
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


def day_count(df: pd.DataFrame) -> pd.DataFrame:
    format_str = r'%Y-%m-%d %H:%M:%S'
    days = []
    for i in df.index:
        if i == 0:
            first_entry = datetime.strptime(df.iloc[i]['TIME'], format_str)
            days.append(1)
        else:
            current_entry = datetime.strptime(df.iloc[i]['TIME'], format_str)
            diff_days = (current_entry - first_entry).days
            days.append(diff_days + 1)
    df['DAY'] = days
    return df


def get_gold(df: pd.DataFrame) -> pd.DataFrame:
    day_groups = {day: group for day, group in df.groupby('DAY')}

    days = []
    pns = []

    for day, group in day_groups.items():
        if len(group[group['IS_NOTE'] != 0]):
            progress_note = group[group['IS_NOTE'] == 1].iloc[-1]['TEXT']
            days.append(day)
            pns.append(progress_note)
    return pd.DataFrame(zip(days, pns), columns=['DAY', 'TEXT'])


def main():
    pd.options.mode.chained_assignment = None  # Turns off the warning

    target_path = 'data/target_population/filtered'

    icu_df = pd.read_csv(f'{target_path}/filtered_ICUSTAYS.csv')

    # edit notes to just be phys progress notes
    notes = pd.read_csv(f'{target_path}/filtered_NOTEEVENTS.csv')
    phys = notes[notes['CATEGORY'] == 'Physician ']
    prog = phys[phys['DESCRIPTION'].str.contains(r'\bProgress Note\b', case=False)]
    

    input_cv = pd.read_csv(f'{target_path}/filtered_INPUTEVENTS_CV.csv')
    input_mv = pd.read_csv(f'{target_path}/filtered_INPUTEVENTS_MV.csv')
    input_mv = input_mv.rename(columns={'STARTTIME': 'CHARTTIME'})

    lab_df = pd.read_csv(f'{target_path}/filtered_LABEVENTS.csv')
    chart_df = pd.read_csv(f'{target_path}/filtered_CHARTEVENTS.csv')
    meds_df = pd.read_csv(f'{target_path}/filtered_PRESCRIPTIONS.csv')

    lab_items = pd.read_csv('data/MIMIC-III/D_LABITEMS.csv')
    chart_items = pd.read_csv('data/MIMIC-III/D_ITEMS.csv')

    output_dir = 'assessment_plan/input'
    gt_dir = 'assessment_plan/gold'

    # get list of admissions that contain physician progress notes - note all do
    prog_ids = prog['HADM_ID'].unique()
    icu_prog = icu_df[icu_df['HADM_ID'].isin(prog_ids)]
    admission_id_list = icu_prog['HADM_ID'].to_list()

    for admission_id in tqdm(admission_id_list):
        dbsource = icu_df[icu_df['HADM_ID'] == admission_id]['DBSOURCE'].values[0]
        if dbsource == 'carevue':
            input_df = input_cv
        else:
            input_df = input_mv

        tab_data = (lab_df, chart_df, input_df, meds_df)
        dictionaries = (lab_items, chart_items)

        struc_tl = get_structured(admission_id, tab_data, dictionaries)
        tl_prog = get_prog_notes(admission_id, prog)
        notes_tl = get_ehr_notes(admission_id, notes)

        combined = pd.concat([struc_tl, tl_prog, notes_tl])
        combined = combined.dropna(subset='TIME')
        combined = combined.sort_values(by='TIME').reset_index(drop=True)
        combined['IS_NOTE'] = combined['IS_NOTE'].apply(lambda x: 1 if x == 1 else 0)

        combined_w_days = day_count(combined)
        combined_tl_rel = get_rel_times(combined_w_days)
        gold_notes = get_gold(combined_w_days)

        combined_tl_rel.to_csv(f'{output_dir}/input_{admission_id}.csv', index=False)
        gold_notes.to_csv(f'{gt_dir}/gt_{admission_id}.csv', index=False)

if __name__ == '__main__':
    main()
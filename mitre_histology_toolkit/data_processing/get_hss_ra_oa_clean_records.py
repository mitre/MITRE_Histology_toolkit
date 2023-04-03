import pandas as pd
import os

# Read and process
ext_path = 'data/external/cleaned_oa_ra_data_mitre4.csv'
ext_df = pd.read_csv(ext_path, dtype = object)

id_dict = {}
for project in ['HSS_RA', 'HSS_OA']:
    id_dict[project] = []
    nd_path = f'data/processed/nuclei/{project}'
    for file in os.listdir(nd_path):
        if ('nuclei' not in file) and ('.csv' in file):
            file_base = file.replace('.csv', '')
            id_dict[project] += [file_base]

id_map = []
for row in range(ext_df.shape[0]):
    proj_list = f'HSS_{ext_df["Project"][row]}'
    study_id = ext_df["Study ID"][row]
    found = False
    for filebase in id_dict[proj_list]:
        filecheck = filebase.split('_')[0]
        if filecheck[0] == '0':
            filecheck = filecheck[1:]
        if int(study_id) == int(filecheck):
            found = True
            id_map += [filebase]
            break
    if not found:
        id_map += [None]

ext_df['map_id'] = id_map
ext_df.to_csv(ext_path, index = False)

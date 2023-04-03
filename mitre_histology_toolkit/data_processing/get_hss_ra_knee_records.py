import pandas as pd
import os

# Read and process
project = 'HSS_RA'
nd_path = f'data/processed/nuclei/{project}'
ext_path = 'data/external/HSSFLARE_PathologyCheck_DATA_LABELS_2020-02-07.csv'
ext_df = pd.read_csv(ext_path, dtype = object)
nd_files = os.listdir(nd_path)

ext_ids = []
for i in range(ext_df.shape[0]):
    if 'Knee' in ext_df['Operative Joint'][i]:
        ext_ids += [ext_df['Study ID'][i]]

ext_set = set(ext_ids)

pat_map = {}
ext_map = {}
for file in nd_files:
    if ('nuclei' not in file) and ('.csv' in file):
        file_base = file.replace('.csv', '')
        pat_map[file_base] = file_base.split('_')[0]
        tt = file_base.split('_')[0]
        if tt in ext_map:
            ext_map[tt] += [file_base]
        else:
            ext_map[tt] = [file_base]

pat_ids = []
for pat_id in ext_set:
    pat_ids += ext_map[pat_id]

pat_ids = sorted(pat_ids)
write_ids = [i + '\n' for i in pat_ids]
write_ids[-1] = write_ids[-1][:-1]
f1 = open('data/external/HSS_RA_knee_patients.txt', 'w')
f1.writelines(write_ids)
f1.close()


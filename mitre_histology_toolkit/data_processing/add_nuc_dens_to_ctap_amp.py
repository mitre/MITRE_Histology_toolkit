import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

os.chdir('C:/Users/dfrezza/Documents/SpyderWorkspace/ai-histology')
### Read excel file and apply preprocessing

# patient_data contains metadata on patient demographics and other clinical factors
patient_data = pd.read_excel('data/external/AMP_CTAP_donor_mapping.xlsx', engine='openpyxl')

# remove new lines from dataframe and column header
patient_data = patient_data.replace('\n',' ', regex=True)
col_names = patient_data.columns.tolist()
col_names = [sub.replace('\n', ' ') for sub in col_names]
patient_data.columns = col_names
patient_data

# create resolution dict
slide_files = os.listdir('data/raw/slides/AMP/')
resolution_map = {}
quick_map = {'svs': 0.5, 'czi': 0.22, 'vsi': 0.35}
for i in slide_files:
    if '.' in i:
        sub_id, suffix = i.split('.')
        sub_id = sub_id.split('_')[0].split(' ')[0]
        resolution_map[sub_id] = quick_map[suffix]

resolution_map =   {'300-0173': 0.5,
                    '300-0174': 0.5,
                    '300-0175': 0.5,
                    '300-0176': 0.5,
                    '300-141': 0.35,
                    '300-141RE': 0.35,
                    '300-142': 0.35,
                    '300-143': 0.35,
                    '300-144': 0.35,
                    '300-145': 0.35,
                    '300-146': 0.35,
                    '300-147': 0.35,
                    '300-148': 0.35,
                    '300-149': 0.35,
                    '300-149B': 0.35,
                    '300-150': 0.35,
                    '300-151': 0.35,
                    '300-154-1': 0.5,
                    '300-168': 0.5,
                    '300-170': 0.5,
                    '300-171': 0.5,
                    '300-172': 0.5,
                    '300-181': 0.35,
                    '300-182': 0.35,
                    '300-183': 0.35,
                    '300-186': 0.35,
                    '300-187': 0.35,
                    '300-1879': 0.22,
                    '300-1880': 0.22,
                    '300-1881': 0.22,
                    '300-211': 0.35,
                    '300-212': 0.35,
                    '300-213': 0.35,
                    '300-214': 0.35,
                    '300-215': 0.35,
                    '300-217': 0.35,
                    '300-218': 0.35,
                    '300-219': 0.35,
                    '300-221': 0.35,
                    '300-223': 0.35,
                    '300-225': 0.35,
                    '300-226': 0.35,
                    '300-228': 0.35,
                    '300-229': 0.35,
                    '300-230': 0.35,
                    '300-231': 0.35,
                    '300-232': 0.35,
                    '300-233': 0.35,
                    '300-234': 0.35,
                    '300-235': 0.35,
                    '300-237': 0.35,
                    '300-248': 0.35,
                    '300-249': 0.35,
                    '300-250': 0.35,
                    '300-251': 0.35,
                    '300-251RE': 0.35,
                    '300-258': 0.35,
                    '300-2598': 0.35,
                    '300-2599': 0.35,
                    '300-2600': 0.35,
                    '300-2601': 0.35,
                    '300-2623': 0.35,
                    '300-2657': 0.35,
                    '300-2658': 0.35,
                    '300-2659': 0.35,
                    '300-2660': 0.35,
                    '300-2661': 0.35,
                    '300-2662': 0.35,
                    '300-2663': 0.35,
                    '300-2665': 0.35,
                    '300-2666': 0.35,
                    '300-2667': 0.35,
                    '300-2668': 0.35,
                    '300-2669': 0.35,
                    '300-2671': 0.35,
                    '300-301': 0.35,
                    '300-302': 0.35,
                    '300-303': 0.35,
                    '300-306': 0.35,
                    '300-307': 0.35,
                    '300-308': 0.35,
                    '300-309': 0.35,
                    '300-310': 0.35,
                    '300-311': 0.35,
                    '300-312': 0.35,
                    '300-313': 0.35,
                    '300-392': 0.35,
                    '300-398': 0.35,
                    '300-407': 0.35,
                    '300-408': 0.35,
                    '300-410': 0.35,
                    '300-411': 0.35,
                    '300-413': 0.35,
                    '300-414': 0.35,
                    '300-415': 0.35,
                    '300-416': 0.35,
                    '300-417': 0.35,
                    '300-418': 0.35,
                    '300-456': 0.35,
                    '300-460': 0.35,
                    '300-461': 0.35,
                    '300-462': 0.35,
                    '300-462RE': 0.35,
                    '300-463': 0.35,
                    '300-464': 0.35,
                    '300-464RE': 0.35,
                    '300-465': 0.35,
                    '300-466': 0.35,
                    '300-467': 0.35,
                    '300-468': 0.35,
                    '300-469': 0.35,
                    '300-470': 0.35,
                    '300-471': 0.35,
                    '300-488': 0.22,
                    '300-489': 0.22,
                    '300-490': 0.22,
                    '300-491': 0.22,
                    '300-496': 0.22,
                    '300-501': 0.22,
                    '300-502': 0.22,
                    '300-503': 0.22,
                    '300-504': 0.22,
                    '300-506': 0.22,
                    '300-507': 0.22,
                    '300-508': 0.22,
                    '300-546': 0.35,
                    '300-547': 0.35}

### Load nuclei density data

# Define analysis hyperparameters
min_nuclei = 10
min_tissue = 0.25

# Read and process
nd_path = 'data/processed/output/AMP'
dtfe_path = 'data/processed/node_features/dtfe/AMP'
nd_files = os.listdir(nd_path)

nuclei_density_res = []
for file in nd_files:
    if 'nuclei' not in file:
        try:
            file_base = file.replace('.csv', '')
            nd_df = pd.read_csv(f'{nd_path}/{file}')
            nloc_df = pd.read_csv(f'{nd_path}/{file_base}_nuclei_pos.csv')
            dtfe_df = pd.read_csv(f'{dtfe_path}/{file_base}_base_dtfe.csv')

            # Apply thresholds
            subject_id = file.split('.csv')[0].split('_')[0]
            tile_area = (1024 * resolution_map[subject_id] * 1e-3)**2
            nd_df = nd_df[nd_df['tissue'] > min_tissue]
            nd_df = nd_df[nd_df['num_nuclei'] > min_nuclei]
            nd_df['tissue_area'] = nd_df['tissue'] * tile_area
            nd_df['nuclei_density'] = nd_df['num_nuclei'] / nd_df['tissue_area']
            
            valid_nuclei = nd_df[['col','row']].merge(nloc_df)[['nuclei_x_wsi', 'nuclei_y_wsi']].round(1)
            dtfe_df['nuclei_x_wsi'] = dtfe_df['nuclei_x_wsi'].round(1)
            dtfe_df['nuclei_y_wsi'] = dtfe_df['nuclei_y_wsi'].round(1)
            #dtfe_df = dtfe_df[dtfe_df.dtfe > 0.001].reset_index(drop = True)
            valid_dtfe = valid_nuclei.merge(dtfe_df[['nuclei_x_wsi', 'nuclei_y_wsi', 'dtfe']])
            
            cluster_key_file = f'data/processed/clusters/AMP/{file_base}_dtfe_cluster_key.csv'
            if os.path.isfile(cluster_key_file):
                clst_df = pd.read_csv(cluster_key_file)
                clst_df['nuclei_x_wsi'] = clst_df['nuclei_x_wsi'].round(1)
                clst_df['nuclei_y_wsi'] = clst_df['nuclei_y_wsi'].round(1)
                valid_clst = valid_dtfe.merge(clst_df[['nuclei_x_wsi', 'nuclei_y_wsi']])
                number_of_clusters = len(np.unique(clst_df.cluster_dtfe_x))
            else:
                #  create empty data frame with correct columns
                valid_clst = valid_dtfe[valid_dtfe.dtfe > 100]
                number_of_clusters = 0
            
            res = {
                'file_name': file,
                'subject_id': subject_id,
                'avg_nuclei_density': nd_df['nuclei_density'].mean(),
                'avg_dtfe': valid_dtfe['dtfe'].mean(),
                'avg_cluster_dtfe': valid_clst['dtfe'].mean(),
                'var_nuclei_density': nd_df['nuclei_density'].var(),
                'var_dtfe': valid_dtfe['dtfe'].var(),
                'var_cluster_dtfe': valid_clst['dtfe'].var(),
                'number_of_clusters': number_of_clusters
            }    
            nuclei_density_res += [res]
        except:
            print(file)
    
nuclei_density_df = pd.DataFrame.from_dict(nuclei_density_res)

### Merge dataframes for reporting
aa = set(patient_data['subject_id'])
bb = set(nuclei_density_df['subject_id'])
cc = aa.difference(bb)
dd = bb.difference(aa)
ab = aa.intersection(bb)

nuc_map = {}
ee = set()
for i in dd:
    j = i.replace('-', '-0')
    if j in aa:
        nuc_map[i] = j
    elif j + 'V0' in aa:
        nuc_map[i] = j + 'V0'
    else:
        ee.add(i)

ff = ab.union(set([i for i in nuc_map.values()]))
gg = aa.difference(ff)

merge_col = []
for i in nuclei_density_df['subject_id']:
    if i in nuc_map:
        merge_col += [nuc_map[i]]
    else:
        merge_col += [i]

nuclei_density_df['merge_col'] = merge_col

col_names = ['avg_nuclei_density', 'var_nuclei_density', 'avg_dtfe', 'var_dtfe', 'avg_cluster_dtfe', 'var_cluster_dtfe', 'number_of_clusters', 'merge_col']
data = pd.merge(patient_data, nuclei_density_df[col_names], 'left', 
                left_on='subject_id', right_on='merge_col').drop(columns = 'merge_col')

data.to_csv('data/external/AMP_CTAP_donor_mapping_nuclei_density_dtfe.csv', index = False)

###############################################################################

orderx = ['E + F + M', 'F', 'T + F', 'T + M', 'T + B', 'M']
sns.boxplot(x = 'new_class', y = 'avg_nuclei_density', hue = 'new_class',
            data = data, palette = 'Dark2', order = orderx, hue_order = orderx)
plt.show()
sns.boxplot(x = 'new_class', y = 'var_nuclei_density', hue = 'new_class',
            data = data, palette = 'Dark2', order = orderx, hue_order = orderx)
plt.show()
sns.boxplot(x = 'new_class', y = 'avg_dtfe', hue = 'new_class',
            data = data, palette = 'Dark2', order = orderx, hue_order = orderx)
plt.show()
sns.boxplot(x = 'new_class', y = 'var_dtfe', hue = 'new_class',
            data = data, palette = 'Dark2', order = orderx, hue_order = orderx)
plt.show()
sns.boxplot(x = 'new_class', y = 'avg_cluster_dtfe', hue = 'new_class',
            data = data, palette = 'Dark2', order = orderx, hue_order = orderx)
plt.show()
sns.boxplot(x = 'new_class', y = 'var_cluster_dtfe', hue = 'new_class',
            data = data, palette = 'Dark2', order = orderx, hue_order = orderx)
plt.show()
sns.boxplot(x = 'new_class', y = 'number_of_clusters', hue = 'new_class',
            data = data, palette = 'Dark2', order = orderx, hue_order = orderx)
plt.show()






data = data.drop_duplicates()
data = data.sort_values(by='file_name')
data = data.reset_index()
data = data.drop(columns=['index', 'nid', 'cid', 'match_score'])
data.to_csv('../reports/AMP_analysis_052021.csv', index=False)
pd.set_option('display.max_rows', 200)
display(data)

### Merge dataframes

aa = pd.merge(match_results, nuclei_density_df, left_on='nid', right_on='subject_id')
data = pd.merge(aa, patient_data_df, left_on='cid', right_on='Subject ID')
data = data.drop_duplicates()
data = data.sort_values(by='file_name')
data = data.reset_index()
pd.set_option('display.max_rows', 200)
display(data)

### How many scenes or samples does each patient have?
counts = data['cid'].value_counts()
counts

### How does nuclei density vary for samples of the same patient?
plt.figure(figsize=(14, 6))
   
sns.boxplot(x="cid", y="avg_nuclei_density", data=data)
sns.swarmplot(x="cid", y="avg_nuclei_density", data=data, color=".25")
plt.xlabel('Subject ID')
plt.ylabel('Nuclei Density [nuclei count per $mm^2$ of tissue]')

### Average nuclei density vs Krenn Inflam
data_sub = data[['cid','avg_nuclei_density', 'Krenn_inflam_avg', 'Krenn_lini_avg']]
temp = data_sub.groupby('cid').mean()
temp.plot.scatter(x='Krenn_inflam_avg', y='avg_nuclei_density')
plt.ylabel('Nuclei Density [nuclei count per $mm^2$ of tissue]')
plt.xlabel('Krenn Inflammation Score]')

g = sns.boxplot(x="Krenn_inflam_avg", y="avg_nuclei_density", data=data)
g = sns.swarmplot(x="Krenn_inflam_avg", y="avg_nuclei_density", data=data, color=".25")
plt.xlabel('Average Krenn Inflammation Score')
plt.ylabel('Nuclei Density [nuclei count per $mm^2$ of tissue]')
plt.title('AMP')
plt.ylim([0,12000])
g.axes.set_xticklabels([i.__dict__['_text'][:4] for i in g.axes.get_xticklabels()])
plt.xticks(rotation=45)
plt.tight_layout()

data['Rounded_Krenn_avg'] = data['Krenn_inflam_avg'].round()
sns.boxplot(x="Rounded_Krenn_avg", y="avg_nuclei_density", data=data)
sns.swarmplot(x="Rounded_Krenn_avg", y="avg_nuclei_density", data=data, color=".25")
plt.xlabel('Rounded Average Krenn Inflammation Score')
plt.ylabel('Nuclei Density [nuclei count per $mm^2$ of tissue]')
plt.title('AMP')
plt.ylim((0, 12000))




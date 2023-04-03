import pandas as pd
import os

# Define analysis hyperparameters
min_nuclei = 10
min_tissue = 0.25
resolution = 0.4964 # microns per pixel

# Read and process
for project in ['HSS_RA', 'HSS_OA']:
    nd_path = f'data/processed/nuclei/{project}'
    dtfe_path = f'data/processed/node_features/dtfe/base/{project}'
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
                tile_area = (1024 * resolution * 1e-3)**2
                nd_df = nd_df[nd_df['tissue'] > min_tissue]
                nd_df = nd_df[nd_df['num_nuclei'] > min_nuclei]
                nd_df['tissue_area'] = nd_df['tissue'] * tile_area
                nd_df['nuclei_density'] = nd_df['num_nuclei'] / nd_df['tissue_area']
                
                valid_nuclei = nd_df[['col','row']].merge(nloc_df)[['nuclei_x_wsi', 'nuclei_y_wsi']].round(1)
                dtfe_df['nuclei_x_wsi'] = dtfe_df['nuclei_x_wsi'].round(1)
                dtfe_df['nuclei_y_wsi'] = dtfe_df['nuclei_y_wsi'].round(1)
                #dtfe_df = dtfe_df[dtfe_df.dtfe > 0.001].reset_index(drop = True)
                valid_dtfe = valid_nuclei.merge(dtfe_df[['nuclei_x_wsi', 'nuclei_y_wsi', 'dtfe']])
                
                res = {
                    'file_name': file,
                    'subject_id': subject_id,
                    'avg_nuclei_density': nd_df['nuclei_density'].mean(),
                    'var_nuclei_density': nd_df['nuclei_density'].var(),
                    'avg_dtfe': valid_dtfe['dtfe'].mean(),
                    'var_dtfe': valid_dtfe['dtfe'].var()
                }    
                nuclei_density_res += [res]
            except:
                print(file)
    
    nuclei_density_df = pd.DataFrame.from_dict(nuclei_density_res)
    nuclei_density_df.to_csv(f'data/external/{project}_nuclei_density.csv', index = False)
    



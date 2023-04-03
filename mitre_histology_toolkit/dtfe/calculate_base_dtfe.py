from . import pydtfe
import pandas as pd
import yaml
import os

def run_dtfe(scene_name, path_to_tile_csv, path_to_nuclei_csv, path_to_output_csv, metadata, tissue_threshold = 0.25):    
    # read in nuclei data and keep only nuclei with sufficient tissue
    tileDF = pd.read_csv(path_to_tile_csv)
    nucDF = pd.read_csv(path_to_nuclei_csv)
    
    combDF = pd.merge(nucDF, tileDF, 'inner', on = ['row','col', 'tile_size'])
    df = combDF[combDF['tissue'] > tissue_threshold][['nuclei_x_wsi', 'nuclei_y_wsi']]
    
    # update positions based on resolution    
    # get dtfe using spatial only
    tot_areas, num_triangles = pydtfe.spec_dtfe2d(df.nuclei_x_wsi * metadata['resolution'][0], 
                                                  df.nuclei_y_wsi * metadata['resolution'][1])
    df['dtfe_area'] = tot_areas
    df['dtfe_numTriangles'] = num_triangles
    df['dtfe'] = num_triangles / tot_areas
        
    dtfe_dir = os.path.dirname(path_to_output_csv)
    if not os.path.exists(dtfe_dir):
        os.makedirs(dtfe_dir)
    
    df.to_csv(path_to_output_csv, index = False)
    return

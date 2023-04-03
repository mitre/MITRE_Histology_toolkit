import pandas as pd
import numpy as np
import cv2
from copy import deepcopy
from . import cell_seg
from scipy.ndimage import label
import ast


def run_cell_seg_experiments(path_to_slide_svs, path_to_nuclei_csv, path_to_output_csv , path_to_output_npz, 
                             path_to_tiss_seg_experiments,path_to_output_fdata):
    """      
        This takes in a txt file of experiments which alter the features and dimensionality reduction mehtod of the tissue segmentation algorithm. 
        The rows of the txt file should be experiments and the columns should be the different parameters. The function does not return anything but writes 
        to file information about the length of heterogenous borders and number of islands of tissue classifications for each experiment. It also writes to file
        the tissue masks. 
             
        Arguments
        ----------
        path_to_slide_svs : str
            path to raw image 
        path_to_nuclei_csv : str
            path to csv with information about nuclei centroid, area, eccentricity, etc.
        path_to_output_csv : str
            path to save csv that contains information about borders and islands and other relevant info
        path_to_output_npz : str
            path to save spixel class mask 
        path_to_tiss_seg_experiments : str
            path to txt file of experiment parameters
        path_to_output_fdata:
            path to save spixel feature data for each experiment
               
        Returns
        ---------
        None
    """
    
    tissue_res = {}
    changes_sum = {}
    masks = []
    island_sums = {}
    s = [[1,1,1],
     [1,1,1],
     [1,1,1]]
    experiments_df = pd.read_csv(path_to_tiss_seg_experiments)
    # iterates over the experiemnts
    for index,row in experiments_df.iterrows():
        #if experiment requires nuclei information it will read in the nuclei dataset
        if row['use_nuc_features']:
            nuc_pos = pd.read_csv(path_to_nuclei_csv)
        else:
            nuc_pos = None
        #run the slic and gmm algorithm for tissue segmentation
        tissue_res.update({row['exper'] : cell_seg.get_tissues(path_to_slide_svs, spixel_size_baseMag=64*64,MAG=4,
                                          use_pca = row['pca'], pca_comps = 3, 
                                          use_umap=row['umap'], umap_comps = 3,
                                          use_rgb=row['use_rbg'],
                                          use_texture = row['use_texture'],
                                          use_gradient= row['use_gradient'],
                                          keep_feats = ast.literal_eval(row['keep_feats']) ,
                                          nuclei_df=nuc_pos, use_nuc_features= row['use_nuc_features'],
                                          nuc_features=['nuclei_width', 'nuclei_height', 'area','eccentricity',
                                                        'major_axis_length','minor_axis_length','solidity'],
                                          nuc_features_moments = ['mean', 'stds', 'ranges'])})
        
        # grab the experiment just run
        tiss = tissue_res[row['exper']]
        img_info = tiss['image'].get_info()
        final_mask = np.zeros((img_info['sizeY'], img_info['sizeX']),dtype = 'uint8')

        islands = []

        for tissue_piece in tiss['tissue_pieces']:
            

            # make spixel classification mask
            def f(x):
                return tissue_piece.fdata['cluster'][x] if x > 0 else 0
            mask_mapped = np.vectorize(f)(tissue_piece.spixel_mask).astype('uint8')
            
            # resize mask to original magnification resolution
            label_mask = cv2.resize(mask_mapped, dsize=(tissue_piece.xmax - tissue_piece.xmin,
                                                          tissue_piece.ymax - tissue_piece.ymin),
                                      interpolation=cv2.INTER_NEAREST)
            
            final_mask[tissue_piece.ymin:tissue_piece.ymax,tissue_piece.xmin:tissue_piece.xmax] = label_mask

            
            for i in np.delete(np.unique(mask_mapped),0):
                tmp = deepcopy(mask_mapped)
                tmp[tmp != i] = 0
                _, num_islands = label(tmp, structure=s)
                islands.append(num_islands)
        
        island_sums.update({row['exper']: islands})
        
        exp = row['exper']
        # store masks for future analysis            
        masks.append(final_mask.astype('uint8'))    
        # count the length of hetergenous borders 
        changes_sum.update( {row['exper']: np.sum(cell_seg.count_heterogenous_border(label_mask))} )
    
    #write masks to npz for all experiments
    np.savez_compressed(f'{path_to_output_npz}', *masks)
    

    
    # take the resized spixel mask fo the most recent experiement (this will be the same across experiemnts upto relabelling)
    
    f_spx_mask = np.zeros((img_info['sizeY'], img_info['sizeX']),dtype = 'uint8')
    fdata_list = []

    for tis_num, tissue_piece in enumerate(tiss['tissue_pieces']):
        # resize mask to original magnification resolution
        re_spx = cv2.resize(tissue_piece.spixel_mask, dsize=(tissue_piece.xmax - tissue_piece.xmin,
                                                      tissue_piece.ymax - tissue_piece.ymin),
                                  interpolation=cv2.INTER_NEAREST) 
        
        f_spx_mask[tissue_piece.ymin:tissue_piece.ymax,tissue_piece.xmin:tissue_piece.xmax] = re_spx
        
        tissue_piece.fdata['tissue_piece'] = tis_num
        fdata_list.append(tissue_piece.fdata)

    pd.concat(fdata_list).to_csv(f'{path_to_output_fdata}',index = True, index_label = 'spx_id')
    # take the border data dictionary and turn it into a df with experiment id
    df_tmp = pd.Series(changes_sum).rename_axis(['exper']).reset_index(name='border_sums')
    # add total length of spixel borders
    df_tmp['tot_border'] = np.sum(cell_seg.count_heterogenous_border(f_spx_mask))
    #add number of pixels in tissue
    df_tmp['num_px'] = (f_spx_mask > 0).sum()
    # add number of spixels
    df_tmp['num_spx'] = tissue_piece.fdata.shape[0]
    # merge border info with island info
    df_tmp = pd.merge(df_tmp, pd.Series(island_sums).rename_axis(['exper']).reset_index(name='islands'))
    
    
    df_tmp.to_csv(f'{path_to_output_csv}', index = False)
    
    del re_spx
    del changes_sum
    

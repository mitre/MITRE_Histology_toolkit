"""
Goal:
    Find the optimal clusters for discriminating between OA and RA images.
Measures:
    The percent of cells contained in aggregates.
    The size of the largest aggregate. (# of cells included or area)
Cluster Parameters:
    DTFE threshold
    Number of cells required
    Edge length
"""
import mitre_multifractal_toolkit.box_counting as bc
#import numpy as bc
from mitre_histology_toolkit.dtfe import process_nuclei_functions as pnf
from mitre_histology_toolkit.dtfe import concave_hull_functions as chf
from matplotlib import pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import yaml
import os

image_name = '75_s0'#'007_SYN_207635_s0'
with open("slurm/configs/HSS_RA_param_config.yaml", "r") as yamlfile:
    configs = yaml.load(yamlfile, Loader = yaml.FullLoader)

path_to_masks = f'{configs["nuclei_directory"]}/{configs["project"]}/{image_name}'
path_to_tiles = f'{configs["tile_directory"]}/{configs["project"]}/{image_name}'
path_to_nuclei = f'{configs["nuclei_directory"]}/{configs["project"]}/{image_name}_nuclei_pos.csv'
path_to_dtfe = f'{configs["dtfe_directory"]}/{configs["project"]}/{image_name}_base_dtfe.csv'
output_directory = f'{configs["cluster_directory"]}/{configs["project"]}'

threshold_list = configs["clustering_loop"]["dtfe_threshold_list"]
edge_range_list = configs["clustering_loop"]["edge_range_list"]
number_nodes_necessary_for_cluster_list = configs["clustering_loop"]["number_nodes_necessary_for_cluster_list"]
inclusion_choice = configs["clustering_loop"]["inclusion_choice"]

tile_size = configs["tiling"]["tile_size"]
overlap = configs["tiling"]["overlap"]


position_precision = 5
nuclei_data = pd.read_csv(path_to_nuclei)
nuclei_data['label_wsi'] = list(range(1, nuclei_data.shape[0] + 1))
wsi_df = pd.read_csv(path_to_dtfe)

pos_to_color = {}
for i in range(len(nuclei_data)):
    pos_to_color[(np.round(nuclei_data.nuclei_x_wsi[i], position_precision), 
                np.round(nuclei_data.nuclei_y_wsi[i], position_precision))] = nuclei_data.label_wsi[i]

compile_wsi_arrays_output = pnf.compile_wsi_arrays(image_name, path_to_masks, 
                                                   path_to_tiles, nuclei_data, 
                                                   tile_size = tile_size, 
                                                   overlap = overlap)

wsi_mask, wsi_tile, wsi_mask_point, min_row, min_col, color_to_label, label_to_color = compile_wsi_arrays_output

###############################################################################
wspdf_list = []
rpdf_list = []
nucdf_list = []
threshold = threshold_list[0]
edge_range = edge_range_list[0]
number_nodes_necessary_for_cluster = number_nodes_necessary_for_cluster_list[0]
# set values for thresholding
for threshold in threshold_list:
    for edge_range in edge_range_list:
        for number_nodes_necessary_for_cluster in number_nodes_necessary_for_cluster_list:
            sub_df = wsi_df[wsi_df['dtfe'] > threshold]
            
            ###############################################################################
            # KDTree for sparse distance calculations
            
            dtfe_graph = pnf.generateDtfeGraph(sub_df, edge_range)
            xpos, ypos, dist, edge_coords = pnf.getPositions(dtfe_graph)
            
            cluster_graph = pnf.reduceClusterGraph(dtfe_graph, number_nodes_necessary_for_cluster)
            
            if cluster_graph.number_of_nodes() == 0:
                has_clusters = False
                xpos_full = sub_df['nuclei_x_wsi'].tolist()
                ypos_full = sub_df['nuclei_y_wsi'].tolist()
            else:
                has_clusters = True
                xpos_full, ypos_full, dist, edge_coords = pnf.getPositions(cluster_graph)
            
            map_to_mask = {}
            for row_ind, array_row in enumerate(wsi_mask):
                for col_ind, array_val in enumerate(array_row):
                    if array_val > 0:
                        if array_val in map_to_mask:
                            map_to_mask[array_val] += [(row_ind, col_ind)]
                        else:
                            map_to_mask[array_val] = [(row_ind, col_ind)]
            
            if has_clusters:
                rp_df, rp_column_names = pnf.calculateClusterProperties(image_name, output_directory, 
                                                       cluster_graph, pos_to_color, 
                                                       map_to_mask, wsi_mask, wsi_tile, 
                                                       inclusion_choice, save = False)
                number_of_clusters = rp_df.shape[0]
            else:
                number_of_clusters = 0
            
            pixel_mask_outputs = pnf.getMaskPixel(xpos_full, ypos_full, pos_to_color, 
                                                  map_to_mask, wsi_mask, wsi_tile, 1, 
                                                  'dtfe')
            
            filtered_mask, filtered_tile, min_bb_row, min_bb_col, max_bb_row, max_bb_col = pixel_mask_outputs
            
            bc_dim_dtfe = bc.boxCount(filtered_mask).fDim
            bc_dim_full = bc.boxCount(wsi_mask).fDim
            
            wspDF = pd.DataFrame([[image_name,
                                   threshold,
                                   edge_range,
                                   number_nodes_necessary_for_cluster,
                                   bc_dim_dtfe,
                                   bc_dim_full,
                                   number_of_clusters,
                                   filtered_mask.sum(),
                                   sub_df.shape[0]]])
            
            wspDF.columns = ['ID', 'dtfe_threshold', 'edge_range', 
                             'number_nodes_necessary_for_cluster', 
                             'bc_frac_dim_dtfe_wsi', 'bc_frac_dim_full_wsi',
                             'num_clusters', 'numPixels', 'numNuclei']
            
            wspdf_list += [wspDF]
            
            if has_clusters:
                concave_hull_areas = []
                nuclei_pos_list = []
                for i, c in enumerate(nx.connected_components(cluster_graph)):
                    xpos, ypos, _, _ = pnf.getPositions(cluster_graph.subgraph(c))
                    pos_df = pd.DataFrame({'x': xpos, 'y': ypos})
                    concave_hull_areas += [chf.getConcaveArea(pos_df)]
                    pos_df['cluster_dtfe_x'] = rp_df['centroid_x'][i]
                    pos_df['cluster_dtfe_y'] = rp_df['centroid_y'][i]
                    pos_df.columns = ['nuclei_x_wsi', 'nuclei_y_wsi', 'cluster_dtfe_x', 'cluster_dtfe_y']
                    nuclei_pos_list += [pos_df]
                
                nuclei_pos_df = pd.concat(nuclei_pos_list)
                nuclei_pos_df['threshold'] = threshold
                nuclei_pos_df['edge_range'] = edge_range
                nuclei_pos_df['number_nodes_necessary_for_cluster'] = number_nodes_necessary_for_cluster
                
                rp_df['concave_area'] = concave_hull_areas
                rp_df['threshold'] = threshold
                rp_df['edge_range'] = edge_range
                rp_df['number_nodes_necessary_for_cluster'] = number_nodes_necessary_for_cluster
                
                nucdf_list += [nuclei_pos_df]
                rpdf_list += [rp_df]

cluster_pref = f'{output_directory}/multiple_params/{inclusion_choice}'
os.makedirs(cluster_pref, exist_ok = True)

wsp_final_df = pd.concat(wspdf_list)
wsp_final_df.to_csv(f'{cluster_pref}/{image_name}_dtfe_wsi.csv', index = False)

if len(nucdf_list) > 0:
    nuc_final_df = pd.concat(nucdf_list)
    nuc_final_df.to_csv(f'{cluster_pref}/{image_name}_dtfe_cluster_key.csv', index = False)

if len(rpdf_list) > 0:
    rp_final_df = pd.concat(nucdf_list)
    rp_final_df.to_csv(f'{cluster_pref}/{image_name}_dtfe_clusters.csv', index = False)
   
print(f'Main file saved to: {cluster_pref}/{image_name}_dtfe_wsi.csv')

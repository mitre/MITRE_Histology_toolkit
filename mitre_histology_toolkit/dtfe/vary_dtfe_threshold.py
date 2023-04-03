from . import process_nuclei_functions as pnf
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def vary_dtfe_threshold(image_name, path_to_masks, path_to_tiles, path_to_nuclei, path_to_dtfe, output_directory, edge_range = 75, number_nodes_necessary_for_cluster = 75, thresholds = np.linspace(1e-3, 1e-2, 30), showPlots = False):
    nuclei_data = pd.read_csv(path_to_nuclei)
    nuclei_data['label_wsi'] = list(range(1, nuclei_data.shape[0] + 1))
    wsiDF = pd.read_csv(path_to_dtfe)

    if showPlots:
        fig = plt.figure(figsize = (6,6))
        plt.scatter(nuclei_data.nuclei_x_wsi, nuclei_data.nuclei_y_wsi, s = 1)
        plt.xticks([])
        plt.yticks([])
        fig.axes[0].set_aspect(1)
        fig.axes[0].invert_yaxis()
        plt.show()
    
    
    cluster_proportion_details = []
    for threshold in thresholds:
        subDF = wsiDF[wsiDF['dtfe'] > threshold]
    
        if showPlots:
            fig = plt.figure(figsize = (6,6))
            plt.scatter(wsiDF.nuclei_x_wsi, wsiDF.nuclei_y_wsi, s = 1, c = wsiDF.dtfe, cmap = 'jet', vmax = threshold)
            plt.xticks([])
            plt.yticks([])
            fig.axes[0].set_aspect(1)
            fig.axes[0].invert_yaxis()
            plt.show()
            
            fig = plt.figure(figsize = (6,6))
            plt.scatter(subDF.nuclei_x_wsi, subDF.nuclei_y_wsi, s = 1)
            plt.xticks([])
            plt.yticks([])
            fig.axes[0].set_aspect(1)
            fig.axes[0].invert_yaxis()
            plt.show()
        
        if subDF.shape[0] > 0:
            ###############################################################################
            # KDTree for sparse distance calculations
            
            dtfe_graph = pnf.generateDtfeGraph(subDF, edge_range)
            xpos, ypos, dist, edge_coords = pnf.getPositions(dtfe_graph)
            
            if showPlots:
                fig = plt.figure(figsize = (6,6))
                plt.scatter(xpos, ypos, s = 1)
                plt.xticks([])
                plt.yticks([])
                fig.axes[0].set_aspect(1)
                fig.axes[0].invert_yaxis()
                plt.show()
            
            cluster_graph = pnf.reduceClusterGraph(dtfe_graph, number_nodes_necessary_for_cluster)
            
            if cluster_graph.number_of_nodes() == 0:
                has_clusters = False
                xpos_full = subDF['nuclei_x_wsi'].tolist()
                ypos_full = subDF['nuclei_y_wsi'].tolist()
            else:
                has_clusters = True
                xpos_full, ypos_full, dist, edge_coords = pnf.getPositions(cluster_graph)
            
            if showPlots and has_clusters:
                xpos, ypos, dist, edge_coords = pnf.getPositions(cluster_graph)
                fig = plt.figure(figsize = (6,6))
                plt.scatter(xpos, ypos, s = 1)
                plt.xticks([])
                plt.yticks([])
                fig.axes[0].set_aspect(1)
                fig.axes[0].invert_yaxis()
                plt.show()
            
            cluster_proportion_details += [[threshold, cluster_graph.number_of_nodes(), subDF.shape[0], wsiDF.shape[0], edge_range, number_nodes_necessary_for_cluster]]
        else:
            cluster_proportion_details += [[threshold, 0, subDF.shape[0], wsiDF.shape[0], edge_range, number_nodes_necessary_for_cluster]]
    
    cpDF = pd.DataFrame(cluster_proportion_details)
    cpDF.columns = ['Threshold', 'Number_Nuclei_Clustered', 'Number_Nuclei_Threshold', 'Total_Nuclei', 'Hyper_Param_Edge', 'Hyper_Param_Quantity']
    
    outName = f'{output_directory}/{image_name}_dtfe_threshold_{edge_range}_{number_nodes_necessary_for_cluster}.csv'
    print(outName)
    cpDF.to_csv(outName, index = False)

# cpDF['Prop'] = cpDF['Number_Nuclei_Clustered'] / cpDF['Total_Nuclei']
# plt.plot(cpDF['Threshold'], cpDF['Prop'])
# plt.plot(cpDF['Threshold'], cpDF['Number_Nuclei_Threshold'])
# plt.plot(cpDF['Threshold'], cpDF['Number_Nuclei_Clustered'] / cpDF['Number_Nuclei_Threshold'])
# plt.plot(cpDF['Threshold'], cpDF['Number_Nuclei_Clustered'] / cpDF['Number_Nuclei_Clustered'].max())
# plt.plot(cpDF['Threshold'], cpDF['Number_Nuclei_Threshold'] / cpDF['Number_Nuclei_Threshold'].max())

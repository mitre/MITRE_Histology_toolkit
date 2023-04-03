from matplotlib.collections import LineCollection
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np


img = 'data/processed/node_features/dtfe/AMP/300-489'
for scene in range(10):
    data = pd.read_csv(f'{img}_s{scene}_base_dtfe.csv')
    fig, ax = plt.subplots()
    ax.scatter(data.nuclei_x_wsi, data.nuclei_y_wsi, s = 1, c = data.dtfe, 
               cmap = 'jet', vmax = 0.002)
    ax.set_title(f'Scene {scene}')
    ax.set_aspect(1)
    ax.invert_yaxis()
    plt.show()



data = pd.read_csv(f'{img}_s1_base_dtfe.csv')
fig, ax = plt.subplots()
ax.scatter(data.nuclei_x_wsi, data.nuclei_y_wsi, s = 1, c = data.dtfe, 
           cmap = 'jet', vmax = 0.002)
ax.set_title(f'Scene {scene}')
ax.set_aspect(1)
xmin, xmax = (9000, 11500)
ymin, ymax = (12500, 15000)
ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
ax.invert_yaxis()


new_data = data[ymin:ymax, xmin:xmax]

def generateDtfeGraph(position_df, edge_range):
    X = np.array(position_df[['nuclei_x_wsi', 'nuclei_y_wsi']])
    kd = KDTree(X)
    kd_net = kd.query_ball_tree(kd, edge_range)
    
    edge_coords = []
    node_list = []
    weighted_edge_list = []
    node_remove_list = []
    for node_ind, edge_list in enumerate(kd_net):
        if len(edge_list) > 3:
            node_list += [(node_ind, {'x': X[node_ind, 0], 'y': X[node_ind, 1]})]
            for ni2 in edge_list:
                if ni2 > node_ind:
                    dba = [X[node_ind], X[ni2]]
                    edge_coords += [dba]
                    weighted_edge_list += [(node_ind, ni2, float(calcDist(dba)))]
        else:
            node_remove_list += [node_ind]
    
    edge_remove_list = []
    for e_id in range(len(weighted_edge_list)):
        if weighted_edge_list[e_id][0] in node_remove_list or weighted_edge_list[e_id][1] in node_remove_list:
            edge_remove_list += [e_id]
    
    edge_remove_list.reverse()
    for e_id in edge_remove_list:
        del weighted_edge_list[e_id]
        del edge_coords[e_id]
    
    edge_coords = np.array(edge_coords)
    
    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_weighted_edges_from(weighted_edge_list)
    
    G = nx.algorithms.tree.mst.minimum_spanning_tree(G)
    return(G)

def reduceClusterGraph(dtfe_graph, number_nodes_necessary_for_cluster):
    S = [dtfe_graph.subgraph(c).copy() for c in nx.connected_components(dtfe_graph)]
    node_rm = []
    for s in S:
        if s.number_of_nodes() < number_nodes_necessary_for_cluster:
            node_rm += list(s.nodes())
    
    cluster_graph = dtfe_graph.copy()
    cluster_graph.remove_nodes_from(node_rm)
    return(cluster_graph)

def calcDist(dba):
    return(np.sqrt((dba[0][0] - dba[1][0])**2 + (dba[0][1] - dba[1][1])**2))

def getPositions(G):
    xpos = [node[1]['x'] for node in G.nodes(data=True)]
    ypos = [node[1]['y'] for node in G.nodes(data=True)]
    dist = [edge[-1]['weight'] for edge in G.edges(data=True)]
    
    xdict = nx.get_node_attributes(G, 'x')
    ydict = nx.get_node_attributes(G, 'y')
    
    edge_coords = []
    for edge in G.edges:
        n1 = [xdict[edge[0]], ydict[edge[0]]]
        n2 = [xdict[edge[1]], ydict[edge[1]]]
        edge_coords += [[n1, n2]]
    
    edge_coords = np.array(edge_coords)
    
    return(xpos, ypos, dist, edge_coords)

def plotNetwork(xpos, ypos, edge_coords, figsize = None):
    if figsize is None:
        figsize = (16,16)
    lines = LineCollection(edge_coords, color = 'blue', linewidth = 0.5)
    fig, ax = plt.subplots(figsize = figsize)
    ax.add_artist(lines)
    ax.plot(xpos, ypos, 'k.')
    return(fig, ax)



wsi_df = data[(data.nuclei_x_wsi > xmin) & (data.nuclei_x_wsi < xmax) & (data.nuclei_y_wsi > ymin) & (data.nuclei_y_wsi < ymax)].reset_index(drop = True)

img_nuc = 'data/processed/output/AMP/300-489_s1_nuclei_pos.csv'
nuclei_data = pd.read_csv(img_nuc)
    
temp_list = [[0 for i in range(xmax - xmin)] for j in range(ymax - ymin)]
xy_list = (np.round(np.array(wsi_df[['nuclei_x_wsi', 'nuclei_y_wsi']])) - np.array([xmin, ymin])).astype(int)
for xi, yi in xy_list:
    temp_list[yi][xi] = 1

wsi_tile = np.array(temp_list)
#wsi_mask, wsi_tile, wsi_mask_point, min_row, min_col, color_to_label, label_to_color = compile_wsi_arrays_output


threshold = 0.002
edge_range = 50
number_nodes_necessary_for_cluster = 75

sub_df = wsi_df[wsi_df['dtfe'] > threshold]

dtfe_graph = generateDtfeGraph(sub_df, edge_range)
xpos, ypos, dist, edge_coords = getPositions(dtfe_graph)

fig = plt.figure(figsize = (6,6))
plt.scatter(xpos, ypos, s = 1)
plt.title('Original')
plt.xticks([])
plt.yticks([])
fig.axes[0].set_aspect(1)
fig.axes[0].invert_yaxis()
plt.show()

cluster_graph = reduceClusterGraph(dtfe_graph, number_nodes_necessary_for_cluster)

xpos, ypos, dist, edge_coords = getPositions(cluster_graph)
fig = plt.figure(figsize = (6,6))
plt.scatter(xpos, ypos, s = 1)
plt.title('Reduced')
plt.xticks([])
plt.yticks([])
fig.axes[0].set_aspect(1)
fig.axes[0].invert_yaxis()
plt.show()




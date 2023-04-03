from skimage import morphology, measure, draw
from . import fractal_dim_functions as fdf
from scipy import ndimage, sparse, spatial
from matplotlib import pyplot as plt
from matplotlib import collections
import networkx as nx
import pandas as pd
import numpy as np
import skimage.io
import os

def compile_wsi_arrays(image_name, mask_path, tile_path, nuclei_data, 
                       tile_size = 1024, overlap = 0):
    """
    Create numpy arrays based on the outputs of the nuclei detection process.

    Parameters
    ----------
    image_name : str
        The name of the whole slide image in the slides directory.
    mask_path : str
        The path to the mask directory for this particular whole slide image.
    tile_path : str
        The path to the tile directory for this particular whole slide image.
    nuclei_data : pd.DataFrame
        The dataframe with tile and nuclei position information.
    tile_size : int, optional
        The edge size of the tiles (in pixels). The default is 1024.
    overlap : int, optional
        The overlap of the tiles (in pixels). The default is 0.

    Returns
    -------
    wsi_mask - np.array of all the mask tiles patched together.
    wsi_tile - np.array of all the tiles patched together.
    wsi_mask_point - np.array of all the mask tiles patched together where each
        nuclei is only marked by its centroid pixel.
    min_row - the minimum row tile to analyze.
    min_col - the minimum column tile to analyze.
    color_to_label - a dict mapping between the marker_color and the row, col, 
        and nuclei label for each nuclei.
    label_to_color - a dict mapping that is the inverse of color_to_label.

    """
    min_row = nuclei_data['row'].min()
    max_row = nuclei_data['row'].max()
    min_col = nuclei_data['col'].min()
    max_col = nuclei_data['col'].max()
    
    
    color_to_label = {}
    label_to_color = {}
    for row_idx, row in enumerate(np.arange(min_row, max_row+1)):
        for col_idx, col in enumerate(np.arange(min_col, max_col+1)):   
            mask_fname = f'{image_name}__R{row}_C{col}_TS{tile_size}_OL{overlap}_nuclei_mask.npz'
            tile_fname = mask_fname.replace('_nuclei_mask.npz', '.jpg')
            
            try:
                tile = skimage.io.imread(f'{tile_path}/{tile_fname}')
                im_nuclei_seg_mask = sparse.load_npz(f'{mask_path}/{mask_fname}').toarray()
                sub_nuclei_data = nuclei_data[(nuclei_data.row == row) & (nuclei_data.col == col)]
                sub_nuclei_data.sort_values('label', inplace = True)
                sub_nuclei_data.reset_index(drop = True, inplace = True)
                mask_point = np.zeros(im_nuclei_seg_mask.shape)
                label_to_color[(row, col)] = {}
                label_to_color[(row, col)][0] = 0
                label_to_color[(row, col)][1] = 0
                for i in range(sub_nuclei_data.shape[0]):
                    if sub_nuclei_data.label[i] > 1:
                        marker_count = sub_nuclei_data['label_wsi'][i]
                        color_to_label[marker_count] = (row, col, sub_nuclei_data.label[i])
                        label_to_color[(row, col)][sub_nuclei_data.label[i]] = marker_count
                        mask_point[int(sub_nuclei_data.nuclei_y_tile[i]), int(sub_nuclei_data.nuclei_x_tile[i])] = marker_count
                
                im_nuclei_seg_mask_mapped = np.array([label_to_color[(row, col)][label_ind] for array_row in im_nuclei_seg_mask for label_ind in array_row]).reshape(im_nuclei_seg_mask.shape)
            
            except:
                im_nuclei_seg_mask_mapped = np.zeros([tile_size, tile_size]).astype(int)
                tile = (np.ones([tile_size, tile_size, 3]) * 255).astype(int)
                mask_point = im_nuclei_seg_mask_mapped.copy()
            
            if col_idx == 0:
                mask_col = im_nuclei_seg_mask_mapped
                tile_col = tile
                mask_point_col = mask_point
            else:
                mask_col = np.concatenate([mask_col, im_nuclei_seg_mask_mapped], axis = 1)
                tile_col = np.concatenate([tile_col, tile], axis = 1)
                mask_point_col = np.concatenate([mask_point_col, mask_point], axis = 1)
        
        if row_idx == 0:
            wsi_mask = mask_col
            wsi_tile = tile_col
            wsi_mask_point = mask_point_col
        else:
            wsi_mask = np.concatenate([wsi_mask, mask_col], axis=0)
            wsi_tile = np.concatenate([wsi_tile, tile_col], axis=0)
            wsi_mask_point = np.concatenate([wsi_mask_point, mask_point_col], axis=0)
    
    return(wsi_mask, wsi_tile, wsi_mask_point, min_row, min_col, color_to_label, label_to_color)

def generate_dtfe_graph(position_df, edge_range):
    """
    Generates a graph using dtfe thresholded nuclei and a KDTree. The edge
    lengths are filtered using the edge_range parameter, given in the same
    units as the position dataframe.

    Parameters
    ----------
    position_df : data frame
        A pandas data frame with columns: ['nuclei_x_wsi', 'nuclei_y_wsi'].
    edge_range : int
        The maximum edge length between associated nuclei. Must be in the same
        units as the positions in position_df.

    Returns
    -------
    A networkx graph with subgraphs for each candidate aggregate.

    """
    points = np.array(position_df[['nuclei_x_wsi', 'nuclei_y_wsi']])
    kd = spatial.KDTree(points)
    kd_net = kd.query_ball_tree(kd, edge_range)
    
    edge_coords = []
    node_list = []
    weighted_edge_list = []
    node_remove_list = []
    for node_ind, edge_list in enumerate(kd_net):
        if len(edge_list) > 3:
            node_list += [(node_ind, {'x': points[node_ind, 0], 'y': points[node_ind, 1]})]
            for ni2 in edge_list:
                if ni2 > node_ind:
                    dba = [points[node_ind], points[ni2]]
                    edge_coords += [dba]
                    weighted_edge_list += [(node_ind, ni2, float(calc_dist(dba)))]
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
    
    graph = nx.Graph()
    graph.add_nodes_from(node_list)
    graph.add_weighted_edges_from(weighted_edge_list)
    
    graph = nx.algorithms.tree.mst.minimum_spanning_tree(graph)
    return(graph)

def reduce_aggregate_graph(dtfe_graph, number_nodes_necessary_for_aggregate):
    """
    Reduces a networkx graph object that holds a set of subgraphs corresponding
    to candidate aggregates. Each subgraph is checked fo quantity of nodes and
    is compared to the number_nodes_necessary_for_aggregate parameter.

    Parameters
    ----------
    dtfe_graph : networkx.Graph
        A graph of subgraphs containing aggregate candidates.
    number_nodes_necessary_for_aggregate : int
        The minimum number of nuclei required to constitute an aggregate.

    Returns
    -------
    A graph of subgraphs corresponding to detected aggregates.

    """
    subgraph = [dtfe_graph.subgraph(c).copy() for c in nx.connected_components(dtfe_graph)]
    node_rm = []
    for sg in subgraph:
        if sg.number_of_nodes() < number_nodes_necessary_for_aggregate:
            node_rm += list(sg.nodes())
    
    aggregate_graph = dtfe_graph.copy()
    aggregate_graph.remove_nodes_from(node_rm)
    return(aggregate_graph)

def calc_dist(dba):
    """
    Calculates the distance between two points.

    Parameters
    ----------
    dba : tuple
        A tuple of the form ([x1, y1], [x2, y2]).

    Returns
    -------
    The distance between points.

    """
    return(np.sqrt((dba[0][0] - dba[1][0])**2 + (dba[0][1] - dba[1][1])**2))

def get_positions(graph):
    """
    A helper function to query graphs for position and distance information

    Parameters
    ----------
    graph : networkx.Graph
        The graph to be queried.

    Returns
    -------
    4 lists:
        - x positions
        - y positions
        - distances
        - edges

    """
    xpos = [node[1]['x'] for node in graph.nodes(data=True)]
    ypos = [node[1]['y'] for node in graph.nodes(data=True)]
    dist = [edge[-1]['weight'] for edge in graph.edges(data=True)]
    
    xdict = nx.get_node_attributes(graph, 'x')
    ydict = nx.get_node_attributes(graph, 'y')
    
    edge_coords = []
    for edge in graph.edges:
        n1 = [xdict[edge[0]], ydict[edge[0]]]
        n2 = [xdict[edge[1]], ydict[edge[1]]]
        edge_coords += [[n1, n2]]
    
    edge_coords = np.array(edge_coords)
    
    return(xpos, ypos, dist, edge_coords)

def plot_network(xpos, ypos, edge_coords, figsize = (16,16)):
    """
    Plots a network efficiently using LineCollections.

    Parameters
    ----------
    xpos : list
        The list of x positions from a graph.
    ypos : list
        The list of y positions from a graph.
    edge_coords : list
        The list of edge tuples.
    figsize : tuple, optional
        The size of the figure (width, height). The default is (16, 16).

    Returns
    -------
    The figure and axes objects.

    """
    lines = collections.LineCollection(edge_coords, color = 'blue', linewidth = 0.5)
    fig, ax = plt.subplots(figsize = figsize)
    ax.add_artist(lines)
    ax.plot(xpos, ypos, 'k.')
    return(fig, ax)

def get_mask_pixel(xpos, ypos, pos_to_color, map_to_mask, wsi_mask, wsi_tile, buffer, inclusion_choice):
    """
    Based on the inclusion_choice parameter (which is either 'dtfe', 'convex',
    or 'bbox'), the image and nuclei mask are queried for the given xpos and
    ypos. The relevant mask selections and bounding box parameters are returned.

    Parameters
    ----------
    xpos : list
        The x positions.
    ypos : list
        The y positions.
    pos_to_color : dict
        The dictionary mapping the position to the nucleus label.
    map_to_mask : dict
        The dictionary mapping the label to its list of pixel coords in the mask.
    wsi_mask : array-like 2D
        A 2D numpy array with pixels labeled according to the location of nuclei.
    wsi_tile : array-like 3D
        A 3D numpy array which represents the histological image.
    buffer : int
        A buffer for the edges of the bounding box in pixel units.
    inclusion_choice : str
        This indicates how nuclei in a region are included. Must be 'dtfe', 
        'convex', or 'bbox'.

    Returns
    -------
    ret_mask, ret_tile, min_bb_row, min_bb_col, max_bb_row, max_bb_col.

    """
    if inclusion_choice == 'dtfe':
        return(get_mask_pixel_dtfe(xpos, ypos, pos_to_color, map_to_mask, wsi_mask, wsi_tile, buffer))
    elif inclusion_choice == 'convex':
        return(get_mask_pixel_convex(xpos, ypos, pos_to_color, map_to_mask, wsi_mask, wsi_tile, buffer))
    elif inclusion_choice == 'bbox':
        return(get_mask_pixel_bbox(xpos, ypos, pos_to_color, map_to_mask, wsi_mask, wsi_tile, buffer))
    else:
        return(get_mask_pixel_dtfe(xpos, ypos, pos_to_color, map_to_mask, wsi_mask, wsi_tile, buffer))

def get_mask_pixel_dtfe(xpos, ypos, pos_to_color, map_to_mask, wsi_mask, wsi_tile, buffer):
    """
    The image and nuclei mask are queried for the given xpos and
    ypos. The relevant mask selections and bounding box parameters are returned.
    Only the nuclei in the xpos, ypos lists are considered.

    Parameters
    ----------
    xpos : list
        The x positions.
    ypos : list
        The y positions.
    pos_to_color : dict
        The dictionary mapping the position to the nucleus label.
    map_to_mask : dict
        The dictionary mapping the label to its list of pixel coords in the mask.
    wsi_mask : array-like 2D
        A 2D numpy array with pixels labeled according to the location of nuclei.
    wsi_tile : array-like 3D
        A 3D numpy array which represents the histological image.
    buffer : int
        A buffer for the edges of the bounding box in pixel units.

    Returns
    -------
    ret_mask, ret_tile, min_bb_row, min_bb_col, max_bb_row, max_bb_col.

    """
    xpr = np.round(xpos, 5)
    ypr = np.round(ypos, 5)
    mask_colors = [pos_to_color[(xpr[idx], ypr[idx])] for idx in range(len(xpos))]
    mask_pixels = [item for sublist in [map_to_mask[mask_color] for mask_color in mask_colors] for item in sublist]
    mask_pixels = np.array(mask_pixels)
    # dealing with boundaries
    mask_pixels = mask_pixels[(mask_pixels[:, 1] <= wsi_mask.shape[1] - buffer - 1) & 
                            (mask_pixels[:, 0] <= wsi_mask.shape[0] - buffer - 1)]
    mask_row = mask_pixels[:, 0]
    mask_col = mask_pixels[:, 1]
    min_bb_row = max(mask_row.min(), buffer)
    max_bb_row = mask_row.max()
    min_bb_col = max(mask_col.min(), buffer)
    max_bb_col = mask_col.max()
    mask_col = mask_col[mask_col <= wsi_mask.shape[1] - buffer - 1]# dealing with boundaries
    mask_row = mask_row[mask_row <= wsi_mask.shape[0] - buffer - 1]
    
    ret_mask = np.zeros((max_bb_row - min_bb_row + buffer*2, max_bb_col - min_bb_col + buffer*2)).astype(int)
    ret_mask[mask_row - min_bb_row + buffer, mask_col - min_bb_col + buffer] = 1
    ret_tile = wsi_tile[(min_bb_row-buffer) : (max_bb_row+buffer), (min_bb_col-buffer) : (max_bb_col+buffer)]
    return(ret_mask, ret_tile, min_bb_row, min_bb_col, max_bb_row, max_bb_col)

def get_mask_pixel_convex(xpos, ypos, pos_to_color, map_to_mask, wsi_mask, wsi_tile, buffer):
    """
    The image and nuclei mask are queried for the given xpos and
    ypos. The relevant mask selections and bounding box parameters are returned.
    All the nuclei in the convex hull created by xpos, ypos lists are considered.

    Parameters
    ----------
    xpos : list
        The x positions.
    ypos : list
        The y positions.
    pos_to_color : dict
        The dictionary mapping the position to the nucleus label.
    map_to_mask : dict
        The dictionary mapping the label to its list of pixel coords in the mask.
    wsi_mask : array-like 2D
        A 2D numpy array with pixels labeled according to the location of nuclei.
    wsi_tile : array-like 3D
        A 3D numpy array which represents the histological image.
    buffer : int
        A buffer for the edges of the bounding box in pixel units.

    Returns
    -------
    ret_mask, ret_tile, min_bb_row, min_bb_col, max_bb_row, max_bb_col.

    """
    xpr = np.round(xpos, 5)
    ypr = np.round(ypos, 5)
    mask_colors = [pos_to_color[(xpr[idx], ypr[idx])] for idx in range(len(xpos))]
    mask_pixels = [item for sublist in [map_to_mask[mask_color] for mask_color in mask_colors] for item in sublist]
    mask_pixels = np.array(mask_pixels)
    # dealing with boundaries
    mask_pixels = mask_pixels[(mask_pixels[:, 1] <= wsi_mask.shape[1] - buffer - 1) & 
                            (mask_pixels[:, 0] <= wsi_mask.shape[0] - buffer - 1)]
    mask_row = mask_pixels[:, 0]
    mask_col = mask_pixels[:, 1]
    min_bb_row = max(mask_row.min(), buffer)
    max_bb_row = mask_row.max()
    min_bb_col = max(mask_col.min(), buffer)
    max_bb_col = mask_col.max()
    
    bb_mask = np.where(wsi_mask[(min_bb_row-buffer) : (max_bb_row+buffer), (min_bb_col-buffer) : (max_bb_col+buffer)] > 0, 1, 0)
    dtfe_mask = np.zeros((max_bb_row - min_bb_row + buffer*2, max_bb_col - min_bb_col + buffer*2))
    dtfe_mask[mask_row - min_bb_row + buffer, mask_col - min_bb_col + buffer] = 1
    convex_mask = morphology.convex_hull_image(dtfe_mask)
    ret_mask = (bb_mask * convex_mask).astype(int)
    ret_tile = wsi_tile[(min_bb_row-buffer) : (max_bb_row+buffer), (min_bb_col-buffer) : (max_bb_col+buffer)]
    return(ret_mask, ret_tile, min_bb_row, min_bb_col, max_bb_row, max_bb_col)

def get_mask_pixel_bbox(xpos, ypos, pos_to_color, map_to_mask, wsi_mask, wsi_tile, buffer):
    """
    The image and nuclei mask are queried for the given xpos and
    ypos. The relevant mask selections and bounding box parameters are returned.
    All the nuclei in the bounding box constructed using the xpos, ypos lists 
    are considered.

    Parameters
    ----------
    xpos : list
        The x positions.
    ypos : list
        The y positions.
    pos_to_color : dict
        The dictionary mapping the position to the nucleus label.
    map_to_mask : dict
        The dictionary mapping the label to its list of pixel coords in the mask.
    wsi_mask : array-like 2D
        A 2D numpy array with pixels labeled according to the location of nuclei.
    wsi_tile : array-like 3D
        A 3D numpy array which represents the histological image.
    buffer : int
        A buffer for the edges of the bounding box in pixel units.

    Returns
    -------
    ret_mask, ret_tile, min_bb_row, min_bb_col, max_bb_row, max_bb_col.

    """
    xpr = np.round(xpos, 5)
    ypr = np.round(ypos, 5)
    mask_colors = [pos_to_color[(xpr[idx], ypr[idx])] for idx in range(len(xpos))]
    mask_pixels = [item for sublist in [map_to_mask[mask_color] for mask_color in mask_colors] for item in sublist]
    mask_pixels = np.array(mask_pixels)
    # dealing with boundaries
    mask_pixels = mask_pixels[(mask_pixels[:, 1] <= wsi_mask.shape[1] - buffer - 1) & 
                            (mask_pixels[:, 0] <= wsi_mask.shape[0] - buffer - 1)]
    mask_row = mask_pixels[:, 0]
    mask_col = mask_pixels[:, 1]
    min_bb_row = max(mask_row.min(), buffer)
    max_bb_row = mask_row.max()
    min_bb_col = max(mask_col.min(), buffer)
    max_bb_col = mask_col.max()
    mask_col = mask_col[mask_col <= wsi_mask.shape[1] - buffer - 1]# dealing with boundaries
    mask_row = mask_row[mask_row <= wsi_mask.shape[0] - buffer - 1]
    
    ret_mask = np.where(wsi_mask[(min_bb_row-buffer) : (max_bb_row+buffer), (min_bb_col-buffer) : (max_bb_col+buffer)] > 0, 1, 0).astype(int)
    ret_tile = wsi_tile[(min_bb_row-buffer) : (max_bb_row+buffer), (min_bb_col-buffer) : (max_bb_col+buffer)]
    return(ret_mask, ret_tile, min_bb_row, min_bb_col, max_bb_row, max_bb_col)

def plot_and_save_aggregate_pair(bb_mask, bb_tile, frac_dim, filename, save = True):
    """
    This takes specific mask arrays to plot and saves the figures. The plots
    show the nuclei masks of aggregates side by side with the corresponding
    sections of the histological image.

    Parameters
    ----------
    bb_mask : array-like
        The 2D numpy array of the nuclei mask.
    bb_tile : array-like
        The 3D numpy array of the histological image.
    frac_dim : float
        The fractal dimension of the nuclei mask.
    filename : str
        The output filename for the plot.
    save : bool, optional
        The flag dictating whether or not to save the plots. The default is True.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    cmap = 'viridis'
    ax[0].imshow(bb_mask, cmap = cmap)
    ax[1].imshow(bb_tile, cmap = cmap)
    fig.suptitle(f'Fractal Dimension = {frac_dim:.3f}')
    if save:
        fig.savefig(filename)
    plt.show()
    plt.clf()
    plt.close('all')

def count_nuclei_along_axes(bb_label_mask, rp, padding = 100, show_plots = False):
    """
    Uses the labeled mask of the bounding box around the aggregate in concert
    with the region properties of the aggregate to determine the number of 
    nuclei along each axis. A line with a a width of approximately 5 pixels is 
    used for the intersection determination.

    Parameters
    ----------
    bb_label_mask : numpy array
        The labeled bounding box of the aggregate with a unique integer per 
        nucleus and the background labeled 0.
    rp : skimage.measure._regionprops.RegionProperties
        The region property object of the aggregate as a whole.
    padding : int, optional
        The padding size of the array. The default is 50.
    show_plots : bool, optional
        Whether to plot the mask and axis assigned nuclei. The default is False.

    Returns
    -------
    The number of nuclei counted along each axis: (major, minor).

    """
    label_mask = np.pad(bb_label_mask, padding)
    temp_mask = np.where(label_mask > 0, 1, 0)
    
    y0, x0 = rp.centroid
    y0 += padding
    x0 += padding
    offset = padding * 2 - 2
    max_y, max_x = temp_mask.shape
    max_y -= 5
    max_x -= 5
    minor_x1 = max(min(x0 + np.cos(rp.orientation) * 0.5 * (rp.minor_axis_length + offset), max_x), 1)
    minor_y1 = max(min(y0 - np.sin(rp.orientation) * 0.5 * (rp.minor_axis_length + offset), max_y), 1)
    minor_x2 = max(min(x0 - np.cos(rp.orientation) * 0.5 * (rp.minor_axis_length + offset), max_x), 1)
    minor_y2 = max(min(y0 + np.sin(rp.orientation) * 0.5 * (rp.minor_axis_length + offset), max_y), 1)
    major_x1 = max(min(x0 - np.sin(rp.orientation) * 0.5 * (rp.major_axis_length + offset), max_x), 1)
    major_y1 = max(min(y0 - np.cos(rp.orientation) * 0.5 * (rp.major_axis_length + offset), max_y), 1)
    major_x2 = max(min(x0 + np.sin(rp.orientation) * 0.5 * (rp.major_axis_length + offset), max_x), 1)
    major_y2 = max(min(y0 + np.cos(rp.orientation) * 0.5 * (rp.major_axis_length + offset), max_y), 1)
        
    rr, cc, val = draw.line_aa(int(major_y1)-1, int(major_x1), int(major_y2)-1, int(major_x2))
    temp_mask[rr, cc] = 2
    rr, cc, val = draw.line_aa(int(major_y1)+1, int(major_x1), int(major_y2)+1, int(major_x2))
    temp_mask[rr, cc] = 2
    rr, cc, val = draw.line_aa(int(minor_y1)-1, int(minor_x1), int(minor_y2)-1, int(minor_x2))
    temp_mask[rr, cc] = 3
    rr, cc, val = draw.line_aa(int(minor_y1)+1, int(minor_x1), int(minor_y2)+1, int(minor_x2))
    temp_mask[rr, cc] = 3
    
    
    match_labels = np.unique(label_mask[temp_mask == 2])[1:]
    num_major_axis = len(match_labels)
    if show_plots:
        ovlp_mask = np.where(label_mask > 0, 1, 0)
        ovlp_mask[np.isin(label_mask, match_labels)] = 2
    
    match_labels = np.unique(label_mask[temp_mask == 3])[1:]
    num_minor_axis = len(match_labels)
    if show_plots:
        ovlp_mask[np.isin(label_mask, match_labels)] = 3
        fig, ax = plt.subplots(figsize = (15,15))
        ax.imshow(ovlp_mask)
        ax.plot((minor_x1, minor_x2), (minor_y1, minor_y2), '-w', linewidth=1, alpha = 0.6)
        ax.plot((major_x1, major_x2), (major_y1, major_y2), '-w', linewidth=1, alpha = 0.6)
        ax.plot(x0, y0, '.g', markersize=15)
        ax.set_title(f'Major Axis: {num_major_axis}, Minor Axis: {num_minor_axis}', fontsize = 15)
        plt.show()
        plt.close('all')
        plt.clf()
    return(num_major_axis, num_minor_axis)

def calculate_aggregate_properties(image_name, image_resolution, output_directory, aggregate_graph, pos_to_color, map_to_mask, wsi_mask, wsi_tile, inclusion_choice, save = True):
    """
    This takes the aggregate graph and the image/nuclei data and extracts 
    relevant aggregate features and indexes them. It returns a data frame
    with one row per aggregate and multiple shape and size features.

    Parameters
    ----------
    image_name : str
        The image name ({id}_s{scene_id}).
    image_resolution : tuple
        The number of mcms per pixel edge (width, height).
    output_directory : str
        The directory for the outputs.
    aggregate_graph : networkx.Graph
        The graph of subgraphs for the aggregates.
    pos_to_color : dict
        The dictionary mapping the position to the nucleus label.
    map_to_mask : dict
        The dictionary mapping the label to its list of pixel coords in the mask.
    wsi_mask : array-like 2D
        A 2D numpy array with pixels labeled according to the location of nuclei.
    wsi_tile : array-like 3D
        A 3D numpy array which represents the histological image.
    inclusion_choice : str
        This indicates how nuclei in a region are included. Must be 'dtfe', 
        'convex', or 'bbox'.
    save : bool, optional
        The flag indicating whether to save out the plots and data. 
        The default is True.

    Returns
    -------
    The data frame with one row per aggregate and the shape/size parameters.

    """
    px_to_sq_mcm = image_resolution[0] * image_resolution[1]
    px_to_mcm = np.sqrt((image_resolution[0]**2 + image_resolution[1]**2)/2)
    buffer = 1#20
    rp_list = []
    for i, c in enumerate(nx.connected_components(aggregate_graph)):
        print(f'=========== Aggregate {i+1} ============')
        xpos, ypos, _, _ = get_positions(aggregate_graph.subgraph(c))
        minx = int(min(xpos))
        miny = int(min(ypos))
        bb_mask, bb_tile, min_bb_row, min_bb_col, max_bb_row, max_bb_col = get_mask_pixel(xpos, ypos, pos_to_color, map_to_mask, wsi_mask, wsi_tile, buffer, inclusion_choice)
        minbb_list = [int(min(ypos)), int(min(xpos)), int(max(ypos)), int(max(xpos))]
        temp_frac_dim = fdf.get_box_counting_dimension(bb_mask)
        bc_dim_aggregate = fdf.get_box_counting_dimension(bb_mask)
        if save:
            ss_bb_mask = sparse.csc_matrix(bb_mask)
            sparse.save_npz(f'{output_directory}/{inclusion_choice}/mask_arrays/{image_name}_aggregate_{i+1}.npz', ss_bb_mask)
            filename = f'{output_directory}/{inclusion_choice}/paired_images/{image_name}_aggregate_{i+1}.png'
            plot_and_save_aggregate_pair(bb_mask, bb_tile, temp_frac_dim, filename, save = save)
        
        bb_label_mask, num_nuclei = ndimage.label(bb_mask)
        reg_props = measure.regionprops(bb_mask)
        rp = reg_props[0]
        num_nuclei_major_axis, num_nuclei_minor_axis = count_nuclei_along_axes(bb_label_mask, rp, show_plots = True)
        rpcy, rpcx = rp.centroid
        rp_list += [[image_name, 
                     i+1, 
                     rpcx + minx, 
                     rpcy + miny, 
                     rp.convex_area * px_to_sq_mcm, 
                     rp.eccentricity, 
                     rp.area * px_to_sq_mcm, 
                     num_nuclei, 
                     rp.major_axis_length * px_to_mcm, 
                     rp.minor_axis_length * px_to_mcm,
                     num_nuclei_major_axis,
                     num_nuclei_minor_axis,
                     bc_dim_aggregate] + minbb_list]
    
    rp_df = pd.DataFrame(rp_list)
    column_names = ['image_id', 'aggregate_id', 'centroid_x', 'centroid_y', 'convex_area',
                    'eccentricity', 'nuclei_area', 'num_nuclei', 'major_axis_length', 
                    'minor_axis_length', 'num_nuclei_major_axis',
                    'num_nuclei_minor_axis', 'bc_frac_dim_aggregate', 
                    'bb_min_row', 'bb_min_col', 'bb_max_row', 'bb_max_col']
        
    rp_df.columns = column_names
    return(rp_df)

def create_aggregate_directory_structure(aggregate_dir):
    """
    Creates the directory structure for the aggregate data if it does not
    already exist.

    Parameters
    ----------
    aggregate_dir : str
        The starting point for the subdirectories. This should be something like
        'project_dir/data/processed/aggregates/data_set'.

    Returns
    -------
    None.

    """
    for dirname in ['bbox', 'dtfe', 'convex']:
        os.makedirs(f'{aggregate_dir}/{dirname}/mask_arrays', exist_ok = True)
        os.makedirs(f'{aggregate_dir}/{dirname}/paired_images', exist_ok = True)

    return

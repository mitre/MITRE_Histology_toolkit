from . import process_nuclei_functions as pnf
from . import concave_hull_functions as chf
from . import fractal_dim_functions as fdf
from matplotlib import pyplot as plt
from scipy import spatial
import networkx as nx
import pandas as pd
import numpy as np
import yaml

def filter_nuclei(nuclei_data, wsi_df, eccentricity_threshold, solidity_threshold, area_threshold, precision = 5):
    """
    Filter nuclei by shape parameters like solidity and eccentricity to reduce
    false detections.

    Parameters
    ----------
    nuclei_data : array-like
        The nuclei position data frame output from the nuclei detection.
    wsi_df : array-like
        The dtfe based data frame. Must have position and dtfe data.
    eccentricity_threshold : float
        The upper bound on the eccentricity of acceptable nuclei.
    solidity_threshold : float
        The lower bound on the solidity of acceptable nuclei.
    area_threshold : float
        The upper bound on the area for possible nucleus detections (measured
        in square microns.
    precision : int, optional
        The number of decimals to keep in the position information. 
        The default is 5.

    Returns
    -------
    A wsi_df data frame with only the nuclei that pass the filter.

    """
    wsi_df['nuclei_x_wsi'] = wsi_df['nuclei_x_wsi'].round(precision)
    wsi_df['nuclei_y_wsi'] = wsi_df['nuclei_y_wsi'].round(precision)
    nuclei_data['nuclei_x_wsi'] = nuclei_data['nuclei_x_wsi'].round(precision)
    nuclei_data['nuclei_y_wsi'] = nuclei_data['nuclei_y_wsi'].round(precision)
    wsi_df = wsi_df.merge(nuclei_data, how = 'inner', on = ['nuclei_x_wsi', 'nuclei_y_wsi'])
    return(wsi_df[(wsi_df.eccentricity < eccentricity_threshold) &
                  (wsi_df.solidity > solidity_threshold) &
                  (wsi_df.area < area_threshold)].reset_index(drop = True))
    
def read_nuclei(path_to_nuclei, path_to_dtfe, dtfe_threshold, eccentricity_threshold = None, solidity_threshold = None, area_threshold = None, filter_nuclei_bool = False, show_plots = False):
    """
    A feeder function for the aggregate detection algorithm. It reads and 
    filters nuclei and dtfe data.

    Parameters
    ----------
    path_to_nuclei : str
        The path to the nuclei data files.
    path_to_dtfe : str
        The path to the dtfe data files.
    dtfe_threshold : float
        The dtfe threshold parameter. (mcm^-2)
    eccentricity_threshold : float
        The upper bound on the eccentricity of acceptable nuclei.
    solidity_threshold : float
        The lower bound on the solidity of acceptable nuclei.
    area_threshold : float
        The upper bound on the area for possible nucleus detections (measured
        in square microns.
    filter_nuclei_bool : boolean, optional
        A flag determining whether or not to filter the nuclei. 
        The default is False.
    show_plots : boolean, optional
        A flag determining whether or not to display intermediary plots. 
        The default is False.

    Returns
    -------
    The nuclei data frame, the dtfe data frame, and the subsetted dtfe data 
    frame. The dtfe data frames are the ones filtered if the flag is set to 
    True.

    """
    nuclei_data = pd.read_csv(path_to_nuclei)
    nuclei_data['label_wsi'] = list(range(1, nuclei_data.shape[0] + 1))
    wsi_df = pd.read_csv(path_to_dtfe)
    
    if filter_nuclei_bool:
        wsi_df = filter_nuclei(nuclei_data, wsi_df, eccentricity_threshold, solidity_threshold, area_threshold)
    
    if show_plots:
        fig = plt.figure(figsize = (6,6))
        plt.scatter(nuclei_data.nuclei_x_wsi, nuclei_data.nuclei_y_wsi, s = 1)
        plt.xticks([])
        plt.yticks([])
        fig.axes[0].set_aspect(1)
        fig.axes[0].invert_yaxis()
        plt.show()
    ###############################################################################
    # set values for thresholding
    
    sub_df = wsi_df[wsi_df['dtfe'] > dtfe_threshold]
    
    if show_plots:
        fig = plt.figure(figsize = (6,6))
        plt.scatter(wsi_df.nuclei_x_wsi, wsi_df.nuclei_y_wsi, s = .1, 
                    c = wsi_df.dtfe, cmap = 'jet', vmax = dtfe_threshold)
        plt.xticks([])
        plt.yticks([])
        fig.axes[0].set_aspect(1)
        fig.axes[0].invert_yaxis()
        plt.show()
        
        fig = plt.figure(figsize = (6,6))
        plt.scatter(sub_df.nuclei_x_wsi, sub_df.nuclei_y_wsi, s = .1)
        plt.xticks([])
        plt.yticks([])
        fig.axes[0].set_aspect(1)
        fig.axes[0].invert_yaxis()
        plt.show()
    return(nuclei_data, wsi_df, sub_df)

def compile_arrays(image_name, path_to_masks, path_to_tiles, nuclei_data, tile_size, overlap, position_precision = 5):
    """
    

    Parameters
    ----------
    image_name : str
        The name of the image to be analyzed.
    path_to_masks : str
        The path to the nuclei mask files.
    path_to_tiles : str
        The path to the tile files.
    nuclei_data : DataFrame
        The nuclei position data.
    tile_size : int
        The number of pixels along one edge of a tile.
    overlap : int
        The overlap in the tiling procedure.
    precision : int, optional
        The number of decimals to keep in the position information. 
        The default is 5.

    Returns
    -------
    The output from pnf.compile_wsi_arrays and the pos_to_color dict. The 
    combined output is wsi_mask, wsi_tile, wsi_mask_point, min_row, min_col, 
    color_to_label, label_to_color, and pos_to_color.

    """
    pos_to_color = {}
    for i in range(len(nuclei_data)):
        pos_to_color[(np.round(nuclei_data.nuclei_x_wsi[i], position_precision), 
                    np.round(nuclei_data.nuclei_y_wsi[i], position_precision))] = nuclei_data.label_wsi[i]
        
    compile_wsi_arrays_output = pnf.compile_wsi_arrays(image_name, path_to_masks, 
                                                       path_to_tiles, nuclei_data, 
                                                       tile_size = tile_size, 
                                                       overlap = overlap)
    return(compile_wsi_arrays_output, pos_to_color)

def get_dtfe_graph(wsi_df, sub_df, edge_range, image_resolution, show_plots = False):
    """
    

    Parameters
    ----------
    wsi_df : array-like
        The whole DTFE data frame with position information.
    sub_df : array-like
        The filtered DTFE data frame (by DTFE threshold).
    edge_range : int
        The maximum distance between associated nuclei (micrometers).
    image_resolution : tuple
        The number of mcms per pixel edge (width, height).
    show_plots : boolean, optional
        Whether to plot the graph outcome. The default is False.

    Returns
    -------
    The dtfe graph object and the xy coords, distances,  and edges.

    """
    #  convert edge range threshold from mcm to pixels bc positions are in px coords
    edge_range_px = edge_range / (np.sqrt((image_resolution[0]**2 + image_resolution[0]**2) / 2))
    dtfe_graph = pnf.generate_dtfe_graph(sub_df, edge_range_px)
    xpos, ypos, dist, edge_coords = pnf.get_positions(dtfe_graph)
    
    if show_plots:
        fig = plt.figure(figsize = (6,6))
        plt.scatter(wsi_df.nuclei_x_wsi, wsi_df.nuclei_y_wsi, s = 1)
        plt.scatter(xpos, ypos, s = 1)
        plt.xticks([])
        plt.yticks([])
        fig.axes[0].set_aspect(1)
        fig.axes[0].invert_yaxis()
        plt.show()
    return(dtfe_graph, (xpos, ypos, dist, edge_coords))

def get_aggregate_graph(wsi_df, sub_df, dtfe_graph, number_nuclei_necessary_for_aggregate, show_plots = False):
    """
    Use the #nuclei threshold to reduce a dtfe_graph to a final aggregate
    graph.

    Parameters
    ----------
    wsi_df : array-like
        The whole DTFE data frame with position information.
    sub_df : array-like
        The filtered DTFE data frame (by DTFE threshold).
    dtfe_graph : networkx.Graph
        The graph output from the get_dtfe_graph function.
    number_nuclei_necessary_for_aggregate : int
        The minimum number of nuclei required to constitute an aggregate.
    show_plots : boolean, optional
        Whether to show plots. The default is False.

    Returns
    -------
    The aggregate graph, a boolean dictating the presence of aggregates, and
    the xy coords, distance list, and edges.

    """
    aggregate_graph = pnf.reduce_aggregate_graph(dtfe_graph, number_nuclei_necessary_for_aggregate)
    
    if aggregate_graph.number_of_nodes() == 0:
        has_aggregates = False
        xpos_full = sub_df['nuclei_x_wsi'].tolist()
        ypos_full = sub_df['nuclei_y_wsi'].tolist()
        dist = None
        edge_coords = None
    else:
        has_aggregates = True
        xpos_full, ypos_full, dist, edge_coords = pnf.get_positions(aggregate_graph)
    
    if show_plots and has_aggregates:
        xpos, ypos, dist, edge_coords = pnf.get_positions(aggregate_graph)
        fig = plt.figure(figsize = (6,6))
        plt.scatter(wsi_df.nuclei_x_wsi, wsi_df.nuclei_y_wsi, s = 1)
        plt.scatter(xpos, ypos, s = 1)
        #plt.scatter(rp_df.centroid_x, rp_df.centroid_y, s = 20)
        plt.xticks([])
        plt.yticks([])
        fig.axes[0].set_aspect(1)
        fig.axes[0].invert_yaxis()
        plt.show()
    return(aggregate_graph, has_aggregates, (xpos_full, ypos_full, dist, edge_coords))

def restore_nuclei_in_convex_hull(members, candidates):
    """
    Checks to see if any candidate nuclei should be restored to the aggregate
    outlined by the convex hull of members.

    Parameters
    ----------
    members : array-like
        The coordinates of the nuclei which are members of the aggregate. 
        An Nx2 array with x positions in column 1 and y positions in column 2.
    candidates : array-like
        The coordinates of the nuclei being tested for inclusion into the aggregate.
        An Nx2 array with x positions in column 1 and y positions in column 2.

    Returns
    -------
    A numpy array of the candidates which should be assigned to the aggregate.

    """
    points = np.array(candidates)
    member_points = np.array(members)
    hull = spatial.Delaunay(member_points)
    points = points[np.where((points[:,0] >= member_points[:,0].min()) &
                             (points[:,0] <= member_points[:,0].max()) &
                             (points[:,1] >= member_points[:,1].min()) &
                             (points[:,1] <= member_points[:,1].max()))[0]]
    return(points[hull.find_simplex(points) >= 0])

def update_nuclei_list_inclusion_choice(members, candidates, inclusion_choice):
    """
    Restores nuclei removed during the process based on the inclusion_choice
    variable: convex, dtfe, or bbox.

    Parameters
    ----------
    members : array-like
        The coordinates of the nuclei which are members of the aggregate. 
        An Nx2 array with x positions in column 1 and y positions in column 2.
    candidates : array-like
        The coordinates of the nuclei being tested for inclusion into the aggregate.
        An Nx2 array with x positions in column 1 and y positions in column 2.

    Returns
    -------
    A DataFrame of the candidates which should be assigned to the aggregate 
    with columns 'x' and 'y'.

    """
    if inclusion_choice == 'convex':
        pos_df = pd.DataFrame(restore_nuclei_in_convex_hull(members, candidates))
        pos_df.columns = ['x', 'y']
    elif inclusion_choice == 'bbox':
        points = np.array(candidates)
        points = points[np.where((points[:,0] >= members.x.min()) &
                                 (points[:,0] <= members.x.max()) &
                                 (points[:,1] >= members.y.min()) &
                                 (points[:,1] <= members.y.max()))[0]]
        pos_df = pd.DataFrame(points)
        pos_df.columns = ['x', 'y']
    else:
        pos_df = members
    
    return(pos_df)

def create_map_to_mask(wsi_mask):
    """
    Maps each pixel coordinate in the nuclei mask to the proper nucleus.
    
    Parameters
    ----------
    wsi_mask : numpy array 2D
        The nuclei mask with 0 corresponding to the background and the pixels
        of each nucleus uniquely labeled.

    Returns
    -------
    map_to_mask : dict
        The keys are the nucleus labels and the values are all the pixel xy
        coordinates associated with the particular nucleus.

    """
    map_to_mask = {}
    for row_ind, array_row in enumerate(wsi_mask):
        for col_ind, array_val in enumerate(array_row):
            if array_val > 0:
                if array_val in map_to_mask:
                    map_to_mask[array_val] += [(row_ind, col_ind)]
                else:
                    map_to_mask[array_val] = [(row_ind, col_ind)]
    return(map_to_mask)

def get_aggregate_nuclei(aggregate_graph, nuclei_data, image_resolution, inclusion_choice, rp_df = None):
    """
    Based on inclusion_choice, nuclei within detected aggregates that were
    previously omitted may be restored according to the convex hull or
    bounding box.

    Parameters
    ----------
    aggregate_graph : networkx.Graph
        The final graph of aggregates (output from get_aggregate_graph).
    nuclei_data : data frame
        The nuclei position information.
    image_resolution : tuple
        The image resolution in mcm/px in the form (width, height).
    inclusion_choice : str
        Must be 'convex', 'dtfe', or 'bbox'. Decides how to restore nuclei.
    rp_df : data frame, optional
        A data frame which has the cluster id information that matches the
        aggregate_graph. 

    Returns
    -------
    The concave hull areas for each aggregate and a data frame of the nuclei
    associated with each aggregate.

    """
    concave_hull_areas = []
    nuclei_pos_list = []
    px_to_sq_mcm = image_resolution[0] * image_resolution[1]
    for i, c in enumerate(nx.connected_components(aggregate_graph)):
        xpos, ypos, _, _ = pnf.get_positions(aggregate_graph.subgraph(c))
        pos_df = pd.DataFrame({'x': xpos, 'y': ypos})
        pos_df = update_nuclei_list_inclusion_choice(pos_df, nuclei_data[['nuclei_x_wsi', 'nuclei_y_wsi']], inclusion_choice)
        concave_hull_areas += [chf.get_concave_area(pos_df) * px_to_sq_mcm]
        pos_df.columns = ['nuclei_x_wsi', 'nuclei_y_wsi']
        if rp_df is not None:
            pos_df['centroid_x'] = rp_df['centroid_x'][i]
            pos_df['centroid_y'] = rp_df['centroid_y'][i]
        else:
            rp_df['cluster_id'] = i
        nuclei_pos_list += [pos_df]
    
    nuclei_pos_df = pd.concat(nuclei_pos_list)
    return(concave_hull_areas, nuclei_pos_df)

def aggregate_detection(image_name, path_to_masks, path_to_tiles, path_to_nuclei, 
                        path_to_dtfe, metadata, output_directory, 
                        inclusion_list = ['dtfe', 'convex', 'bbox'],
                        dtfe_threshold = 0.0284, edge_range = 50, 
                        number_nuclei_necessary_for_aggregate = 30, 
                        tile_size = 1024, overlap = 0, filter_nuclei_bool = False,
                        eccentricity_threshold = 0.8, solidity_threshold = 0.9, 
                        area_threshold = 50, show_plots = False, 
                        save_aggregate_data = True):
    """
    Performs the aggregate detection algorithm. 

    Parameters
    ----------
    image_name : str
        The name of the image (following id_s{scene} convention e.g. 45_s0).
    path_to_masks : str
        The path to the nuclei masks.
    path_to_tiles : str
        The path to the tile images.
    path_to_nuclei : str
        The path to the nuclei position data frames.
    path_to_dtfe : str
        The path to the dtfe data frames.
    metadata : dict
        The metadata dict associated with the scene to be analyzed.
    output_directory : str
        The output directory for the aggregate data.
    inclusion_list : list, optional
        The list must contain at least one of ['dtfe', 'convex', 'bbox']. 
        The default is ['dtfe', 'convex', 'bbox'].
    dtfe_threshold : float, optional
        The lower bound dtfe value in (mcm^-2) for aggregate inclusion. 
        The default is 0.0284.
    edge_range : int, optional
        The maximum distance between associated nuclei (mcm). The default is 50.
    number_nuclei_necessary_for_aggregate : int, optional
        The minimum number of nuclei required to constitute an aggregate.
        The default is 30.
    tile_size : int, optional
        The number of pixels on a tile edge. The default is 1024.
    overlap : int, optional
        The size of the pixel overlap in the tiling procedure. The default is 0.
    filter_nuclei_bool : boolean, optional
        Whether to filter the nuclei by shape feature. The default is False.
    eccentricity_threshold : float, optional
        Only used if filter_nuclei_bool == True. The upper bound on the 
        eccentricity of acceptable nuclei. The default is 0.8.
    solidity_threshold : float, optional
        Only used if filter_nuclei_bool == True. The lower bound on the 
        solidity of acceptable nuclei. The default is 0.9.
    area_threshold : float, optional
        Only used if filter_nuclei_bool == True. The upper bound on the area 
        of acceptable nuclei (mcm^2). The default is 50.
    show_plots : boolean, optional
        Whether to show plots. The default is False.
    save_aggregate_data : boolean, optional
        Whether to save out the aggregate outcome data. The default is True.

    Returns
    -------
    None.

    """
    image_resolution = metadata['resolution']
    
    # if output directory structure isn't yet set up, do it now
    if save_aggregate_data:
        pnf.create_aggregate_directory_structure(output_directory)
    
    nuclei_data, wsi_df, sub_df = read_nuclei(path_to_nuclei, path_to_dtfe, dtfe_threshold, eccentricity_threshold = eccentricity_threshold, solidity_threshold = solidity_threshold, area_threshold = area_threshold, filter_nuclei_bool = filter_nuclei_bool, show_plots = show_plots)
    dtfe_graph, (xpos, ypos, dist, edge_coords) = get_dtfe_graph(wsi_df, sub_df, edge_range, image_resolution, show_plots = show_plots)
    aggregate_graph, has_aggregates, (xpos_final, ypos_final, dist_final, edge_coords_final) = get_aggregate_graph(wsi_df, sub_df, dtfe_graph, number_nuclei_necessary_for_aggregate, show_plots = show_plots)
    
    compile_wsi_arrays_output, pos_to_color = compile_arrays(image_name, path_to_masks, path_to_tiles, nuclei_data, tile_size, overlap)
    wsi_mask, wsi_tile, wsi_mask_point, min_row, min_col, color_to_label, label_to_color = compile_wsi_arrays_output
    
    map_to_mask = create_map_to_mask(wsi_mask)
    
    for inclusion_choice in inclusion_list:
        rp_df = pd.DataFrame()
        if has_aggregates:
            rp_df = pnf.calculate_aggregate_properties(image_name, image_resolution,
                                                       output_directory, 
                                                       aggregate_graph, pos_to_color, 
                                                       map_to_mask, wsi_mask, wsi_tile, 
                                                       inclusion_choice, save = save_aggregate_data)
        
            pixel_mask_outputs = pnf.get_mask_pixel(xpos_final, ypos_final, pos_to_color, 
                                                    map_to_mask, wsi_mask, wsi_tile, 1, 
                                                    'dtfe') #dtfe for whole image

            filtered_mask, filtered_tile, min_bb_row, min_bb_col, max_bb_row, max_bb_col = pixel_mask_outputs
            bc_dim_dtfe = fdf.get_box_counting_dimension(filtered_mask)
        else:
            filtered_mask = np.empty(0)
            bc_dim_dtfe = 0
        
        bc_dim_full = fdf.get_box_counting_dimension(wsi_mask)
        
        wsp_df = pd.DataFrame([[image_name,
                               dtfe_threshold,
                               bc_dim_dtfe,
                               bc_dim_full,
                               rp_df.shape[0],
                               filtered_mask.sum() * image_resolution[0] * image_resolution[1],
                               aggregate_graph.number_of_nodes(),
                               sub_df.shape[0]]])
        
        wsp_df.columns = ['image_id', 'dtfe_threshold', 'bc_frac_dim_aggregate_wsi', 
                          'bc_frac_dim_full_wsi', 'num_aggregates', 'nuclei_area_within_aggregates', 
                          'num_nuclei_in_aggregates', 'num_nuclei_above_dtfe']
    
        aggregate_pref = f'{output_directory}/{inclusion_choice}/{image_name}'  
        if save_aggregate_data:
            wsp_df.to_csv(f'{aggregate_pref}_dtfe_wsi.csv', index = False)
        
        
        if has_aggregates:
            concave_hull_areas, nuclei_pos_df = get_aggregate_nuclei(aggregate_graph, nuclei_data, image_resolution, inclusion_choice, rp_df)
            rp_df['concave_area'] = concave_hull_areas
            
            if save_aggregate_data:
                nuclei_pos_df.to_csv(f'{aggregate_pref}_dtfe_aggregate_key.csv', index = False)
                rp_df.to_csv(f'{aggregate_pref}_dtfe_aggregates.csv', index = False)
        
        if save_aggregate_data:
            print(f'main file saved to: {aggregate_pref}_dtfe_aggregates.csv')
    return

def aggregate_detection_loop_parameters(image_name, path_to_masks, path_to_tiles, 
                                        path_to_nuclei, path_to_dtfe, 
                                        path_to_metadata, output_directory, 
                                        dtfe_threshold_list, edge_range_list, 
                                        number_nuclei_necessary_for_aggregate_list, 
                                        inclusion_list = ['dtfe', 'convex', 'bbox'],
                                        tile_size = 1024, overlap = 0, 
                                        filter_nuclei_bool = False,
                                        eccentricity_threshold = 0.8, 
                                        solidity_threshold = 0.9, 
                                        area_threshold = 50, show_plots = False, 
                                        save_aggregate_data = True):
    """
    Performs the aggregate detection algorithm looped over lists of hyper
    parameters for the dtfe_threshold, edge_range, and 
    number_nuclei_necessary_for_aggregate.

    Parameters
    ----------
    image_name : str
        The name of the image (following id_s{scene} convention e.g. 45_s0).
    path_to_masks : str
        The path to the nuclei masks.
    path_to_tiles : str
        The path to the tile images.
    path_to_nuclei : str
        The path to the nuclei position data frames.
    path_to_dtfe : str
        The path to the dtfe data frames.
    path_to_metadata : str
        The path to the metadata yaml file.
    output_directory : str
        The output directory for the aggregate data.
    dtfe_threshold_list : list
        The list of lower bound dtfe value in (mcm^-2) for aggregate inclusion. 
    edge_range_list : list
        The list of maximum distances between associated nuclei (mcm). 
    number_nuclei_necessary_for_aggregate_list : list
        The list of minimum numbers of nuclei required to constitute an aggregate.
    inclusion_list : list, optional
        The list must contain at least one of ['dtfe', 'convex', 'bbox']. 
        The default is ['dtfe', 'convex', 'bbox'].
    tile_size : int, optional
        The number of pixels on a tile edge. The default is 1024.
    overlap : int, optional
        The size of the pixel overlap in the tiling procedure. The default is 0.
    filter_nuclei_bool : boolean, optional
        Whether to filter the nuclei by shape feature. The default is False.
    eccentricity_threshold : float, optional
        Only used if filter_nuclei_bool == True. The upper bound on the 
        eccentricity of acceptable nuclei. The default is 0.8.
    solidity_threshold : float, optional
        Only used if filter_nuclei_bool == True. The lower bound on the 
        solidity of acceptable nuclei. The default is 0.9.
    area_threshold : float, optional
        Only used if filter_nuclei_bool == True. The upper bound on the area 
        of acceptable nuclei (mcm^2). The default is 50.
    show_plots : boolean, optional
        Whether to show plots. The default is False.
    save_aggregate_data : boolean, optional
        Whether to save out the aggregate outcome data. The default is True.

    Returns
    -------
    None.

    """
    with open(path_to_metadata, "r") as yamlfile:
        metadata_full = yaml.load(yamlfile, Loader = yaml.FullLoader)
    
    metadata = metadata_full[image_name.split('_s')[-1]]
    image_resolution = metadata['resolution']
    # if output directory structure isn't yet set up, do it now
    if save_aggregate_data:
        pnf.create_aggregate_directory_structure(output_directory)
    
    nuclei_data, wsi_df, sub_df = read_nuclei(path_to_nuclei, path_to_dtfe, 0, eccentricity_threshold = eccentricity_threshold, solidity_threshold = solidity_threshold, area_threshold = area_threshold, filter_nuclei_bool = filter_nuclei_bool, show_plots = show_plots)
    compile_wsi_arrays_output, pos_to_color = compile_arrays(image_name, path_to_masks, path_to_tiles, nuclei_data, tile_size, overlap)
    wsi_mask, wsi_tile, wsi_mask_point, min_row, min_col, color_to_label, label_to_color = compile_wsi_arrays_output
    map_to_mask = create_map_to_mask(wsi_mask)

    for dtfe_threshold in dtfe_threshold_list:
        for edge_range in edge_range_list:
            for number_nuclei_necessary_for_aggregate in number_nuclei_necessary_for_aggregate_list:
                sub_df = wsi_df[wsi_df['dtfe'] > dtfe_threshold]
                dtfe_graph, (xpos, ypos, dist, edge_coords) = get_dtfe_graph(wsi_df, sub_df, edge_range, image_resolution, show_plots = show_plots)
                aggregate_graph, has_aggregates, (xpos_final, ypos_final, dist_final, edge_coords_final) = get_aggregate_graph(wsi_df, sub_df, dtfe_graph, number_nuclei_necessary_for_aggregate, show_plots = show_plots)
                
                for inclusion_choice in inclusion_list:
                    rp_df = pd.DataFrame()
                    if has_aggregates:
                        rp_df = pnf.calculate_aggregate_properties(image_name, image_resolution, 
                                                                   output_directory, 
                                                                   aggregate_graph, pos_to_color, 
                                                                   map_to_mask, wsi_mask, wsi_tile, 
                                                                   inclusion_choice, save = save_aggregate_data)
                    
                        pixel_mask_outputs = pnf.get_mask_pixel(xpos_final, ypos_final, pos_to_color, 
                                                              map_to_mask, wsi_mask, wsi_tile, 1, 
                                                              'dtfe') #dtfe for whole image

                        filtered_mask, filtered_tile, min_bb_row, min_bb_col, max_bb_row, max_bb_col = pixel_mask_outputs
                        bc_dim_dtfe = fdf.get_box_counting_dimension(filtered_mask)
                    else:
                        filtered_mask = np.empty(0)
                        bc_dim_dtfe = 0
        
                    bc_dim_full = fdf.get_box_counting_dimension(wsi_mask)
                    
                    wsp_df = pd.DataFrame([[image_name,
                                           dtfe_threshold,
                                           bc_dim_dtfe,
                                           bc_dim_full,
                                           rp_df.shape[0],
                                           filtered_mask.sum() * image_resolution[0] * image_resolution[1],
                                           aggregate_graph.number_of_nodes(),
                                           sub_df.shape[0]]])
                    
                    wsp_df.columns = ['image_id', 'dtfe_threshold', 'bc_frac_dim_aggregate_wsi', 
                                     'bc_frac_dim_full_wsi', 'num_aggregates', 'nuclei_area_within_aggregates', 'num_nuclei_in_aggregates', 'num_nuclei_above_dtfe']
                
                    aggregate_pref = f'{output_directory}/{inclusion_choice}/{image_name}'  
                    if save_aggregate_data:
                        wsp_df.to_csv(f'{aggregate_pref}_dtfe_wsi_{dtfe_threshold}_{edge_range}_{number_nuclei_necessary_for_aggregate}.csv', index = False)
                    
                    if has_aggregates:
                        concave_hull_areas, nuclei_pos_df = get_aggregate_nuclei(aggregate_graph, nuclei_data, image_resolution, inclusion_choice, rp_df)
                        rp_df['concave_area'] = concave_hull_areas
                        
                        if save_aggregate_data:
                            nuclei_pos_df.to_csv(f'{aggregate_pref}_dtfe_aggregate_key_{dtfe_threshold}_{edge_range}_{number_nuclei_necessary_for_aggregate}.csv', index = False)
                            rp_df.to_csv(f'{aggregate_pref}_dtfe_aggregates_{dtfe_threshold}_{edge_range}_{number_nuclei_necessary_for_aggregate}.csv', index = False)
                        
                    if save_aggregate_data:
                        print(f'main file saved to: {aggregate_pref}_dtfe_wsi_{dtfe_threshold}_{edge_range}_{number_nuclei_necessary_for_aggregate}.csv')
    return

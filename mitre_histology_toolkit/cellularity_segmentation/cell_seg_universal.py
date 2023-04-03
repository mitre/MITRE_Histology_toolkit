from shapely.geometry import Point
from mitre_histology_toolkit.image import image_loader
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
import numpy as np
from . import cellularity_detection_superpixels_universal
from matplotlib.colors import ListedColormap
import os
import cv2
import rtree
import math
import itertools


"""
Example process of running this:

import cell_seg
import pandas as pd

nuclei_pos = pd.read_csv("/path/to/119_svs_nuclei_pos.csv")

tissues = cell_seg.get_tissues("/path/to/119.svs", spixel_size_baseMag=64*64, MAG=3)
final_mask = cell_seg.rescale_mask(tissues)
nuclei_pos2 = cell_seg.nuclei_assignment(nuclei_pos, final_mask)

"""

"""
intensity_features: [
        'Intensity.Min',
        'Intensity.Max',
        'Intensity.Mean',
        'Intensity.Median',
        'Intensity.MeanMedianDiff',
        'Intensity.Std',
        'Intensity.IQR',
        'Intensity.MAD',
        'Intensity.Skewness',
        'Intensity.Kurtosis',
        'Intensity.HistEnergy',
        'Intensity.HistEntropy',
    ]



haralick_features: [
        'Haralick.ASM',
        'Haralick.Contrast',
        'Haralick.Correlation',
        'Haralick.SumOfSquares',
        'Haralick.IDM',
        'Haralick.SumAverage',
        'Haralick.SumVariance',
        'Haralick.SumEntropy',
        'Haralick.Entropy',
        'Haralick.DifferenceVariance',
        'Haralick.DifferenceEntropy',
        'Haralick.IMC1',
        'Haralick.IMC2',
    ]

"""
def get_tissues(filepath, MAG=3.0, compactness=0.1, spixel_size_baseMag=64 * 64, max_cellularity=40,
                visualize_tissue_boundary=False, visualize_spixels=False, visualize_contiguous=False, 
                n_gaussian_components = 5, use_intensity = True, use_texture = False, 
                keep_feats = ["Intensity.Mean", "Intensity.Median", "Intensity.Std",
                              "Intensity.IQR", "Intensity.HistEntropy"], use_db = False, use_gradient = False,
                use_mf = False, 
               use_pca = False, pca_comps = 0, use_umap = False, umap_comps = 0, nuclei_df = None, 
               use_nuc_features = False, nuc_features = None,
                nuc_features_moments = ['mean', 'stds', 'ranges', 'skew', 'kurt'], use_nuclei_count = False,
               use_rgb = False, anno_dir = 'tmp', image_name = 'tmp', tile_dir = '',mask_dir ='',blend_nuc = False):
    """Detect tissues
            This uses the OpenSlide version of the image loader. If the Bioformats version
            of the image loader is needed, replace
                img = image_loader.ImageLoaderOpenSlide(filepath)
            with
                img = image_loader.ImageLoaderBioformat(filepath)
                img.idx = idx # idx is the scene index
        Arguments
        ----------
        filename : string
            path to the whole slide image file 
        MAG : float
            magnification at which to detect cellularity
        compactness : float
            compactness parameter for the SLIC method. Higher values result
            in more regular superpixels while smaller values are more likely
        spixel_size_baseMag : int
            approximate superpixel size at base (scan) magnification
        max_cellularity : int
            Range [0, 100] or None. If None, normalize visualization RGB values
            for each tissue piece separately, else normalize by given number.
        visualize_tissue_boundary : bool
            whether to visualize result from tissue detection component
        visualize_spixels : bool
            whether to visualize superpixels, color-coded by cellularity.
            setting it to false to only return spixelmask
        visualize_contiguous : bool
            whether to visualize contiguous cellular regions
        n_gaussian_components : int
            number of gaussians to use in GMM to assign cellularity cluster
        use_texture : bool
            whether to calculate texture features 
        keep_feats : list 
            The intensity/texture features to incorporate into the GMM. Fewer features may be better.
        use_db : bool
            Whether to use DBSCAN over GMM
        use_gradient : bool 
            Whether to calculate gradient features
            

        Returns
        -------
        dict
            Returned dict has 2 keys: 
                "tissue_pieces" - output from cellularity_detection_superpixels module, 
                "image" - the image loader object
        """
    basename = os.path.basename(filepath).replace(".","_")
    img = image_loader.ImageLoaderOpenSlide(filepath)
    # color map
    vals = np.random.rand(256,3)
    vals[0, ...] = [0.9, 0.9, 0.9]
    cMap = mcolors.ListedColormap(1 - vals)

    cnorm_thumbnail = {
        'mu': np.array([9.24496373, -0.00966569,  0.01757247]),
        'sigma': np.array([0.35686209, 0.02566772, 0.02500282]),
    }
    # from the ROI in Amgad et al, 2019
    cnorm_main = {
        'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
        'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
    }
    
    # init cellularity detector
    cds = cellularity_detection_superpixels_universal.Cellularity_detector_superpixels(
        img,
        MAG=MAG, compactness=compactness, spixel_size_baseMag=spixel_size_baseMag,
        max_cellularity=max_cellularity,visualize_tissue_boundary=visualize_tissue_boundary,
        visualize_spixels=visualize_spixels, visualize_contiguous=visualize_contiguous,
        n_gaussian_components = n_gaussian_components, use_intensity = use_intensity,
        use_texture = use_texture, keep_feats = keep_feats, use_db = use_db, use_gradient = use_gradient,
        use_mf = use_mf,
        use_pca = use_pca, pca_comps = pca_comps, use_umap = use_umap, umap_comps = umap_comps,
        nuclei_df = nuclei_df, use_nuc_features = use_nuc_features, nuc_features = nuc_features,
                nuc_features_moments = nuc_features_moments, use_nuclei_count = use_nuclei_count, use_rgb = use_rgb,
        anno_dir = anno_dir, image_name = image_name,tile_dir = tile_dir,mask_dir = mask_dir,blend_nuc = blend_nuc,
        get_tissue_mask_kwargs={
            'deconvolve_first': False,
            'n_thresholding_steps': 2,
            'sigma': 1.5,
            'min_size': 500, },
        verbose=2, monitorPrefix=basename)

    cds.set_color_normalization_values(
        mu=cnorm_main['mu'], sigma=cnorm_main['sigma'], what='main')
    
    tissue_pieces = cds.run()
    return {"tissue_pieces": tissue_pieces, "image": img}

def rescale_mask(tissues, annotated = False):
    """Rescale spixel mask
        This function maps the cellularity score, rescales the superpixel 
        mask to the original magnification, pad the tissue piece boundary,
        and merge multiple tissue pieces together to create single final mask.

    Arguments
    ----------
    tissues : dictionary
        output of get_tissues(). Contains 2 key/value pair, 'image'
        has the ImageLoader object where we get the original resolution; 
        'tissue_pieces' has the cellularity_detection_superpixels object
    Returns
    -------
    Numpy Array
        Returned 2d array with the final mask
    """
    img_info = tissues['image'].get_info()
    
    final_mask = np.zeros((img_info['sizeY'], img_info['sizeX']),dtype = 'uint8')

    tissue_piece = tissues['tissue_pieces']
    # convert mask to cellularity score
#         def f(x):
#             return tissue_piece.cluster_props[tissue_piece.fdata['cluster'][x]]['cellularity'] if x > 0 else 0
    def f(x):
        return tissue_piece.fdata['cluster'][x] if x > 0 else 0

    mask_mapped = np.vectorize(f)(tissue_piece.spixel_mask).astype('uint8')

    # resize mask to original magnification resolution
    if annotated == True:
        resized_mask = cv2.resize(mask_mapped, dsize=(tissue_piece.cd.xmax - tissue_piece.cd.xmin,
                                                  tissue_piece.cd.ymax - tissue_piece.cd.ymin),
                              interpolation=cv2.INTER_NEAREST) 
        # pad tissue pieces and merge
        final_mask[tissue_piece.cd.ymin:tissue_piece.cd.ymax,tissue_piece.cd.xmin:tissue_piece.cd.xmax] = resized_mask
    else:
        resized_mask = cv2.resize(mask_mapped, dsize=(tissue_piece.xmax - tissue_piece.xmin,
                                                      tissue_piece.ymax - tissue_piece.ymin),
                                  interpolation=cv2.INTER_NEAREST) 
        # pad tissue pieces and merge
        final_mask[tissue_piece.ymin:tissue_piece.ymax,tissue_piece.xmin:tissue_piece.xmax] = resized_mask

    return final_mask
    

        
def nuclei_assignment(nuclei_pos, final_mask, x_col_label = 'wsi_x', y_col_label = 'wsi_y', 
                      rad = 200, k=1, visualize=True):
    """Assign nuclei with cellularity score
        This takes each nuclei position and map the corresponding cellularity score.

    Arguments
    ----------
    nuclei_pos : Pandas Dataframe
        Contains position of each nuclei
    final_mask : np 2d array
        Contains the cellularity score for each pixel
    x_col_label : String
        The label for column in `nuclei_pos` that contains the x coordinate of the nuclei centroid
    y_col_label : String
        The label for column in `nuclei_pos` that contains the y coordinate of the nuclei centroid
    rad : int
        The radius for which the nearest neighbor point must be within of the unassigned point to copy the tissue label
    k : int
        The number of nearest neighbors to consider
    visualize : bool
        Whether to draw the nuclei assignment
    Returns
    -------
    Pandas Dataframe
        Updated dataframe with `tissue_label` column.
    """

    def get_tissue_label(row):
        return final_mask[int(row['tissue_coord_y'])][int(row['tissue_coord_x'])]
    
    def generator_function(data):
        for i, obj in enumerate(data):
            yield (i, obj.bounds, obj)
        
    nuclei_pos['tissue_coord_x'] = np.round(nuclei_pos.nuclei_x_wsi)
    nuclei_pos['tissue_coord_y'] = np.round(nuclei_pos.nuclei_y_wsi)
    nuclei_pos = nuclei_pos.astype({'tissue_coord_x': 'int32', 'tissue_coord_y': 'int32'})
    nuclei_pos['tissue_label'] = nuclei_pos.apply(get_tissue_label, axis=1)
    nuclei_pos.drop(columns=['tissue_coord_x', 'tissue_coord_y'], inplace=True)
    
    assigned_df = nuclei_pos[nuclei_pos['tissue_label'] != 0]
    unassigned_df = nuclei_pos[nuclei_pos['tissue_label'] == 0]
    
    assigned_pts = [Point(x, y) for x,y in zip(assigned_df['nuclei_x_wsi'], assigned_df['nuclei_y_wsi'])]
    unassigned_pts = [Point(x, y) for x,y in zip(unassigned_df['nuclei_x_wsi'], unassigned_df['nuclei_y_wsi'])]
    
    tree  = rtree.index.Index(generator_function(assigned_pts))
    
    for pt in unassigned_pts:
        nearest_pt = assigned_pts[list(tree.nearest(pt.bounds, k, objects = False))[np.random.randint(k)]]
        tis = nuclei_pos.loc[(nuclei_pos['nuclei_x_wsi'] == nearest_pt.x) & (nuclei_pos['nuclei_y_wsi'] == nearest_pt.y), 'tissue_label']    
    
        if math.hypot(nearest_pt.x - pt.x, nearest_pt.y - pt.y) > rad:
            nuclei_pos.loc[(nuclei_pos['nuclei_x_wsi'] == pt.x) & (nuclei_pos['nuclei_y_wsi'] == pt.y), 'tissue_label'] = 0
        else:
            nuclei_pos.loc[(nuclei_pos['nuclei_x_wsi'] == pt.x) & (nuclei_pos['nuclei_y_wsi'] == pt.y), 'tissue_label'] = tis
        
    if visualize:
#         bounds = [0,1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170]
#         colors = ["r", "b", "g", 'y', 'orange','lightsteelblue', 'darkgoldenrod', 'darkkhaki',
#                   'brown', 'violet', 'mediumspringgreen', 'rosybrown' , 'maroon', 'mediumorchid',
#                  'rebeccapurple', 'plum', 'khaki']
#         n = len(np.unique(nuclei_pos['tissue_label'])[~np.isnan(np.unique(nuclei_pos['tissue_label']))])
#         cmap = mcolors.ListedColormap(colors[:n])
#         norm = mcolors.BoundaryNorm(bounds[:n+1], cmap.N)

#         fig, ax = plt.subplots(figsize = (20,35))
#         im = ax.scatter('nuclei_x_wsi','nuclei_y_wsi', c='tissue_label', s=1,data=nuclei_pos,  cmap=cmap, marker='.', norm = norm)
#         ax.set_aspect(1)
#         ax.invert_yaxis()
#         cbar = plt.colorbar(im, spacing = 'proportional', label = 'Cellularity')
#         cbar.ax.tick_params(labelsize=40)
#         fig, ax = plt.subplots(figsize = (20,35))
#         im = ax.scatter('wsi_x','wsi_y',s=1,data=nuclei_pos, c='tissue_label', cmap='Set1',marker='.')
#         ax.invert_yaxis()
#         plt.colorbar(im)
        
        colors = mcolors.ListedColormap(cm.get_cmap("tab20").colors[:]).colors
        fig, ax = plt.subplots(figsize = (24,50))
        for i,clus in enumerate(np.unique(nuclei_pos['tissue_label'].dropna())):
            ax.scatter(nuclei_pos[nuclei_pos['tissue_label']==clus].nuclei_x_wsi,
                       nuclei_pos[nuclei_pos['tissue_label']==clus].nuclei_y_wsi,
                      c = np.array([colors[i]]), marker = '+', s= 1, label = clus);
        ax.set_aspect(1)
        ax.legend(fontsize = '30', markerscale=20,
                   scatterpoints=1, title = 'Cluster', title_fontsize=30)
        ax.invert_yaxis()
        plt.show()

    return nuclei_pos

# https://stackoverflow.com/questions/48248773/numpy-counting-unique-neighbours-in-2d-array

def count_heterogenous_border(ar):
    """ Gets the length of superpixels which border different classes.
    
    Arguments
    ----------
    ar : Numpy 2D Array
        Array which the superpixel classification mask
    Returns
    -------
    Numpy 2D Array
        Array with the number of borders.
    """
    a = np.pad(ar, (1,1), mode='reflect')
    c = a[1:-1,1:-1]

    top = a[:-2,1:-1]
    bottom = a[2:,1:-1]
    left = a[1:-1,:-2] 
    right = a[1:-1,2:]

    ineq = [top!= c,bottom!= c, left!= c, right!= c]
    count = ineq[0].astype(int) + ineq[1] + ineq[2] + ineq[3] 

    blck = [top, bottom, left, right]
    for i,j in list(itertools.combinations(range(4), r=2)):
        count -= ((blck[i] == blck[j]) & ineq[j])
    return count
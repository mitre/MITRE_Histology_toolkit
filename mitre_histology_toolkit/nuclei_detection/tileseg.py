from skimage import morphology, measure, filters, segmentation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import histomicstk as htk
import numpy as np
import cv2

"""
This module contains functions related to processing or analyzing
a single tile.

Key functions include:
    color_deconvolution
    binarize_image
    find_nuclei
    save_seg_image

"""

def optical_density(tile):
    """
    Convert a tile to optical density values.

    Parameters
    ----------
    tile : numpy array
        RGB image.
    """
    tile = tile.astype(np.float64)
    od = -np.log((tile+1)/255 + 1e-8)
    return od

def standardize_image_background(scene, intensity_threshold = 200):
    """
    Accounts for scanning artifacts in the scene background. Attempts to set
    all background pixels to pure white. Useful as a pre-processing step 
    before optical density is calculated and tissue is detected. Also sets
    all pitch black areas to white.

    Parameters
    ----------
    scene : numpy array
        The image scene to be updated (often at a low magnification).
    intensity_threshold : int, optional
        The lowest intensity value for any RGB that indicates a pixel is
        background. The default is 200.

    Returns
    -------
    A numpy array at the same resolution as the input array with the background
    pixels set to (255, 255, 255).

    """
    im_copy = scene.copy()
    im_copy[np.where(im_copy.min(axis = 2) > intensity_threshold)] = (255, 255, 255)
    im_copy[np.where(im_copy.max(axis = 2) == 0)] = (255, 255, 255)
    return(im_copy)

def find_tissue(tile, intensity_threshold = 200):
    """
    Segments the tissue and calculates the proportion of the image contaning tissue

    Parameters
    ----------
    tile : numpy array
        RGB image.
    intensity_threshold : int, optional
        The lowest intensity value for any RGB that indicates a pixel is
        background. The default is 200.
    """
    # Account for scanning artifacts in background pixels
    tile = standardize_image_background(tile, intensity_threshold = intensity_threshold)
    
    # Convert to optical density values
    tile = optical_density(tile)
    
    # Threshold at beta and create binary image
    beta = 0.12
    tile = np.max(tile, axis=2) >= beta

    # Remove small holes and islands in the image
    #tile = binary_opening(tile, morphology.disk(3))
    #tile = binary_closing(tile, morphology.disk(3))

    # Calculate percentage of tile containig tissue
    percentage = np.mean(tile)
    tissue_amount = percentage #>= tissue_threshold

    return(tissue_amount, tile)

def find_tissue_arx(tile):
    """
    Segments the tissue and calculates the proportion of the image contaning tissue

    Parameters
    ----------
    tile : numpy array
        RGB image.
    """

    # Convert to optical density values
    tile = optical_density(tile)

    # Threshold at beta and create binary image
    beta = 0.12
    tile = np.min(tile, axis=2) >= beta

    # Remove small holes and islands in the image
    #tile = binary_opening(tile, morphology.disk(3))
    #tile = binary_closing(tile, morphology.disk(3))

    # Calculate percentage of tile containig tissue
    percentage = np.mean(tile)
    tissue_amount = percentage #>= tissue_threshold

    return tissue_amount, tile

def color_normalization(tile):
    p, mask_out = find_tissue(tile)

    W_target = np.array([
        [0.65, 0.07, 0],
        [0.70, 0.99, 0],
        [0.29, 0.11, 0]
    ])

    stain_unmixing_routine_params = {
        'stains': ['hematoxylin', 'eosin'],
        'stain_unmixing_method': 'macenko_pca',
    }

    tissue_rgb_normalized = htk.preprocessing.color_normalization.deconvolution_based_normalization(
            tile,  W_target=W_target,
            stain_unmixing_routine_params=stain_unmixing_routine_params,
            mask_out=mask_out)

    return(tissue_rgb_normalized, mask_out)

def color_deconvolution(tile):
    """
    Applies color deconvolution to separate the RGB images into grayscale
    images representing the concetration of hematoxylin and eosin.

    Parameters
    ----------
    tile : numpy array
        RGB image.
    """
    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
            'eosin',        # cytoplasm stain
            'null']         # set to null if input contains only two stains

    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    # create stain matrix
    W = np.array([stain_color_map[st] for st in stains]).T

    # Perform color deconvolution
    deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(tile, W)

    im_stains = deconv_result.Stains


    return(im_stains)


def binarize_image(tile, im_nuclei_stain, foreground_threshold, local_radius_ratio=3, minimum_radius = 3):
    """
    Binarizes an image using a combination of local thresholding and bounding
    boxes to identify candidate locations of nuclei.

    Parameters
    ----------
    tile : numpy array
        RGB image.
    im_nuclei_stain : numpy array
        grayscale image.
    foreground_threshold : float
        threshold for separating foreground and background.
    local_radius_ratio : float, optional
        factor expand bounding box. The default is 3.
    minimum_radius : float, optional
        radius of smallest nuclei. The default is 3.
    """
    img_gray_flat = im_nuclei_stain.flatten()
    thresh = [filters.threshold_otsu(img_gray_flat[img_gray_flat < foreground_threshold])]


    ## Apply initial global threshold
    img_bin = np.copy(im_nuclei_stain)
    img_bin[img_bin > thresh[0]] = 0
    img_bin[img_bin > 0] = 1

    img_bin = morphology.remove_small_objects(img_bin.astype(bool), min_size=16)

    ## Identify connected regions("components") in the image
    #regions = cv2.connectedComponents(img_bin)[1]
    regions = measure.label(img_bin, background=0)
    obj_props = measure.regionprops(regions, intensity_image=im_nuclei_stain)

    ## Initialize mask
    im_fgnd_mask = np.zeros(im_nuclei_stain.shape).astype(np.uint8)

    ## Iterate through regions found via global thresholding
    for obj in obj_props:

        # Skip thresholding on background component
        if (obj.label == 0):
            continue

        # Expand bounding box based on local_radius_ratio
        # The idea is to include more background for local thresholding.
        bbox = obj.bbox
        equivalent_diameter = obj.equivalent_diameter
        min_row = np.max([0, np.round(bbox[0] - equivalent_diameter*local_radius_ratio)]).astype(np.int)
        max_row = np.min([tile.shape[0], np.round(bbox[2] + equivalent_diameter*local_radius_ratio)]).astype(np.int)
        min_col = np.max([0, np.round(bbox[1] - equivalent_diameter*local_radius_ratio)]).astype(np.int)
        max_col = np.min([tile.shape[1], np.round(bbox[3] + equivalent_diameter*local_radius_ratio)]).astype(np.int)
        region = im_nuclei_stain[min_row:max_row, min_col:max_col]
        region_flat = region.flatten()

        # If local threshold fail. Default to global threshold instead.
        try:
            thresh = filters.threshold_otsu(region_flat[region_flat<foreground_threshold])
        except:
            thresh = foreground_threshold

        # Copy local bbox mask to larger tile mask
        region_bin = np.copy(region)
        region_bin[region<thresh] = 1
        region_bin[region>=thresh] = 0
        im_fgnd_mask[min_row:max_row, min_col:max_col] = im_fgnd_mask[min_row:max_row, min_col:max_col] + region_bin.astype(np.uint8)
        im_fgnd_mask[im_fgnd_mask>0] = 1
        
    # Fill small holes in the image
    im_fgnd_mask = morphology.remove_small_holes(im_fgnd_mask.astype(bool), area_threshold=24)

    # Remove small structures in the image based on minimum_radius
    im_fgnd_mask = morphology.remove_small_objects(im_fgnd_mask.astype(bool), min_size=16)
    im_fgnd_mask = im_fgnd_mask.astype(np.uint8)


    return(im_fgnd_mask)

def iter_binarization(im_fgnd_mask, im_nuclei_stain, w_min=32, w_max=256):
    # Iterate through large objects and threshold
    iters = 5
    im_bin_local = im_fgnd_mask #initialize for first iteration
    for ii in range(0,iters):
        
        # Remove small objects and holes
        im_objs = morphology.remove_small_holes(im_bin_local.astype(int), int(w_min/2))
        im_objs = morphology.remove_small_objects(measure.label(im_objs), w_min)

        
        objs = measure.regionprops(im_objs, im_nuclei_stain)
        im_bin_local = np.zeros(im_nuclei_stain.shape)

        for obj in objs:
            b = obj['bbox'] 

            if(obj['area'] < w_max):
                im_bin_local[b[0]:b[2], b[1]:b[3]] += obj['image']  
                continue

            obj_mask = morphology.binary_dilation(obj['image'], morphology.disk(1))
            #obj_hull = obj['convex_image']
            #obj_im = obj['intensity_image']
            obj_im = im_nuclei_stain[b[0]:b[2], b[1]:b[3]]

            #Otsu Threshold
            im_flat = obj_im.flatten()
            mask_flat = obj_mask.flatten()
            thresh = [filters.threshold_otsu(im_flat[mask_flat>0])]

            obj_bin = obj_im.copy()
            obj_bin[obj_im > thresh] = 0
            obj_bin[obj_im <= thresh] = 1
            obj_bin = obj_bin * obj_mask

            #Assign mask
            b = obj['bbox']    
            im_bin_local[b[0]:b[2], b[1]:b[3]] += obj_bin
            
        im_bin_local = morphology.remove_small_holes(im_bin_local.astype(int), int(w_min/2))
        im_bin_local = morphology.remove_small_objects(measure.label(im_bin_local), w_min)
        im_bin_local[im_bin_local>0] = 1
        
    im_bin_local = filters.median(im_bin_local, morphology.disk(3))
    return(im_bin_local.astype(np.uint8))


def find_nuclei(tile, im_nuclei_stain, im_fgnd_mask, image_resolution, min_nucleus_area=6):
    """
    Split the binary image of nuclei into a map of individual nuclei

    Parameters
    ----------
    tile : numpy array
        RGB image.
    im_nuclei_stain : numpy array
        grayscale image.
    im_fgnd_mask : numpy array
        binary mask of the foreground.
    image_resolution : tuple
        The image resolution in mcm/px in the form (width, height).
    min_nucleus_area : int, optional
        area of the smallest nuclei (in square mcm). The default is 6.
    """
    #sure_fg_threshold = 0.30

    im_fgnd_mask = im_fgnd_mask.astype(np.uint8)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(im_fgnd_mask,cv2.MORPH_OPEN,kernel, iterations = 1)
    opening = im_fgnd_mask
    # Identify sure background area
    #kernel = np.ones((5,5),np.uint8)
    #sure_bg = cv2.dilate(opening,kernel,iterations=1)
    #sure_bg = im_fgnd_mask

    _ret, objects = cv2.connectedComponents(opening)
    obj_props = measure.regionprops(objects)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    stain_inverse = cv2.bitwise_not(im_nuclei_stain)
    stain_inverse = stain_inverse - np.min(stain_inverse[:])
    stain_inverse = (stain_inverse / np.max(stain_inverse[:])) * 255

    # Iterate through objects found
    sure_fg = np.zeros(im_nuclei_stain.shape)
    for obj in obj_props:
        #obj = obj_props[250]
        bbox = obj.bbox

        # Calculate normalized distance map
        dist = dist_transform[bbox[0]:bbox[2], bbox[1]:bbox[3]] * obj.image
        dist = dist - np.min(dist[:])
        dist = (dist/np.max(dist[:]))*255

        # Normalize image region
        im = stain_inverse[bbox[0]:bbox[2], bbox[1]:bbox[3]]* obj.image
        im = im - np.min(im[:])
        im = (im/np.max(im[:]))*255

        # Combine distance and image then perform thresholding
        combined = (im + dist)/2
        combined_flat = combined.flatten()
        try:
            thresh = filters.threshold_otsu(combined_flat[combined_flat > 0]) * 0.85
        except:
            thresh = np.max(combined[:]) * 0.5
        _ret, temp = cv2.threshold(combined, thresh, 255, 0)

        # Save to sure foreground map
        sure_fg[bbox[0]:bbox[2], bbox[1]:bbox[3]] = temp
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(im_fgnd_mask, sure_fg)

    # Marker measure.labelling
    _ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all measure.labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==1] = 0

    #markers = cv2.watershed(tile,markers)
    markers = segmentation.watershed(im_nuclei_stain, markers = markers, 
                                     watershed_line = True, 
                                     mask = im_fgnd_mask, compactness = 50)
    # measure.label boundary lines as background
    #markers[markers==-1] = 1
    markers[markers==1] = 0
    '''
    # Perform model-based corrections
    markers = model_based_correction(markers, im_fgnd_mask, im_nuclei_stain)

    # Remeasure.label markers
    markers[markers==1] = 0
    markers[markers>1] = 1
    markers = measure.label(markers, connectivity=1) + 1
    '''
    # Remove small objects according to min_nucleus area
    obj_props = measure.regionprops(markers)
    min_nucleus_area_px = min_nucleus_area / (image_resolution[0] * image_resolution[1])
    for obj in obj_props:
        if (obj.area < min_nucleus_area_px):
            markers[markers==obj.label] = 1
    

    obj_props = measure.regionprops(markers, intensity_image = im_nuclei_stain)
    return(markers, obj_props)


def save_seg_image(im_nuclei_stain, obj_props):
    """
    Save image

    Parameters
    ----------
    tile : numpy array
        RGB image.
    im_nuclei_stain : numpy array
        grayscale image.
    """
    plt.figure(figsize=(20, 20))
    plt.imshow(im_nuclei_stain, origin='lower', cmap='gray', vmin=0, vmax=255)
    plt.xlim([0, im_nuclei_stain.shape[1]])
    plt.ylim([0, im_nuclei_stain.shape[0]])
    plt.title('Nuclei bounding boxes', fontsize=24)

    for i, _props in enumerate(obj_props):

        c = [obj_props[i].centroid[1], obj_props[i].centroid[0], 0]
        width = obj_props[i].bbox[3] - obj_props[i].bbox[1] + 1
        height = obj_props[i].bbox[2] - obj_props[i].bbox[0] + 1

        plt.plot(c[0], c[1], 'w+')
        mrect = mpatches.Rectangle([c[0] - 0.5 * width, c[1] - 0.5 * height] ,
                                width, height, fill = False, ec = 'r', linewidth = 1)
        plt.gca().add_patch(mrect)

    #figure_name =
    plt.savefig('Detect_Nuclei.jpg')

def model_based_correction(initial_markers, im_fgnd_mask, im_nuclei_stain, solidity_threshold = 0.85):
    
    refined_markers = np.copy(initial_markers)
    objProps = measure.regionprops(initial_markers, intensity_image = im_nuclei_stain)

    for ii in range(0,len(objProps)): 
        bbox = objProps[ii].bbox
        #eccen = np.round(objProps[ii].eccentricity, 3)
        solidity = np.round(objProps[ii].solidity, 3)
        measure.label = objProps[ii].measure.label
        if solidity < solidity_threshold:          
            rs = np.max([0, bbox[0]])
            re = np.min([1024, bbox[2]])
            cs = np.max([0, bbox[1]])
            ce = np.min([1024, bbox[3]])

            im_mod = np.copy(im_nuclei_stain[rs:re, cs:ce])
            im_mod_bin = np.copy(im_fgnd_mask[rs:re, cs:ce])
            temp = initial_markers[rs:re, cs:ce]
            im_mod_bin[temp != measure.label] = 0

            kernel = np.ones((3,3), np.uint8)

            # sure background area
            sure_bg = cv2.dilate(im_mod_bin, kernel, iterations = 3)

            # Finding sure foreground area
            dist = cv2.distanceTransform(im_mod_bin,cv2.DIST_L2, 5)
            dist = dist - np.min(dist[:])
            dist = (dist / np.max(dist[:])) * 255
            ret, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg,sure_fg)

            # Marker measure.labelling
            ret, markers = cv2.connectedComponents(sure_fg)

            # Add one to all measure.labels so that sure background is not 0, but 1
            markers = markers + 1

            # Now, mark the region of unknown with zero
            markers[unknown == 1] = 0

            # Apply watershed segmentation
            seg = segmentation.watershed(im_mod, markers = markers, 
                                         watershed_line = True, 
                                         mask = im_mod_bin, compactness = 50)

            # Update masks
            idx = np.max(initial_markers)
            refined_markers[refined_markers == measure.label] = 1
            seg = seg + idx
            seg[seg == idx] = 0
            refined_markers[rs:re, cs:ce] = seg + refined_markers[rs:re, cs:ce]
            
    return(refined_markers)
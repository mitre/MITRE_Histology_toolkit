from skimage import morphology, measure
from ..image import image_loader
from skimage import transform
from scipy import ndimage
from . import tileseg
import scipy.sparse
import numpy as np
import skimage.io
import json
import os


"""
This module contains functions related to processing or analyzing
the whole slide image.

Key functions include:
    process_slide
    eval_tile
    detect_nuclei
    slide_to_tile
    process_tiles

"""

## Supporting Functions
def average_intensity(regionmask, intensitymask):
    #Select only pixels in the region for quantification
    a = regionmask.flatten()
    b = intensitymask.flatten()
    c = b[a==1]
    return (int(np.mean(c)))

def std_intensity(regionmask, intensitymask):
    #Select only pixels in the region for quantification
    a = regionmask.flatten()
    b = intensitymask.flatten()
    c = b[a==1]
    return (int(np.std(c)))

def detect_nuclei(tile, image_resolution, foreground_threshold = 140, min_nucleus_area = 6):
    """
    Detect nuclei in a tile from a histological slide.  Uses the HistomicsTK package.
    Works by isolating the nuclei/hematoxlin channel from the image.
    Apparently this stain is correlated with nuclei (read up on this)

    Args:
    tile: a numpy array for a tile in a histological image
    image_resolution: the tuple (width, height) mcm/px
    foreground_threshold:	threshold for non-tissue
    min_radius
    max_radius
    local_max_search_radius
    min_nucleus_area (mcm^2)
    Returns:

    nuclei: dictionaries with the following properties:
    num_nuclei:	 		number of nuclei present
    im_nuclei_seg_mask: segmentation mask for the nuclei
    'objProps':			list of skimage region objects corresponding to bounding boxes for the nuclei

    """

    nuclei = {}

    ### Step 1:  Perform Color Deconvolution
    try:
        tile, mask_out = tileseg.color_normalization(tile)
    except:
        print("Color Normalization Failed")
    im_stains = tileseg.color_deconvolution(tile)

    ### Step 2: Segment the Nuclei
    # get nuclei/hematoxylin channel
    im_nuclei_stain = im_stains[:, :, 0]
    # binarize image
    im_fgnd_mask = tileseg.binarize_image(tile = tile, 
                                          im_nuclei_stain = im_nuclei_stain, 
                                          foreground_threshold = foreground_threshold)
    im_fgnd_mask = tileseg.iter_binarization(im_fgnd_mask, im_nuclei_stain)
    
    # segment image
    im_nuclei_seg_mask, objProps = tileseg.find_nuclei(tile, im_nuclei_stain,
                                                       im_fgnd_mask, image_resolution,
                                                       min_nucleus_area = min_nucleus_area)

    # shape analysis
    euler = np.zeros(im_nuclei_seg_mask.shape).astype(np.int)
    circularity = np.zeros(im_nuclei_seg_mask.shape).astype(np.float)
    for obj in objProps:
        euler[im_nuclei_seg_mask==obj.label] = obj.euler_number
        circularity[im_nuclei_seg_mask==obj.label] = circ(obj)
    im_nuclei_seg_mask[euler < -1] = 1
    im_nuclei_seg_mask[circularity < 0.3] = 1
    objProps = measure.regionprops(im_nuclei_seg_mask, intensity_image = im_nuclei_stain, 
                                   extra_properties = (average_intensity, std_intensity,))
    
    objProps = objProps[1:]
    nuclei['num_nuclei'] = len(objProps)
    nuclei['im_nuclei_seg_mask'] = im_nuclei_seg_mask
    nuclei['objProps'] = objProps
    return(nuclei)

def circ(obj):
    """
    Defines the circularity of an object.

    Parameters
    ----------
    obj : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    area = obj.area
    perimeter = obj.perimeter
    c = (4 * np.pi * area) / (perimeter * perimeter)
    c = np.min([c,1])
    return(c)

def row_based_idx(num_rows, num_cols, idx):
	"""
	Subplots wants to loop over rows but tiles are ordered by coloumns.
	This function gives row based indices
	"""
	return np.arange(1, num_rows*num_cols + 1).reshape((num_rows, num_cols)).transpose().flatten()[idx-1]

def process_tiles(tile_dir,
                  scene_metadata,
                  tile_size = 512,
                  overlap = 0,
                  tissue_eval = True,
                  find_nuclei = True,
                  tissue_thresh = 0.05,
                  foreground_threshold = 140, 
                  min_nucleus_area = 6,
                  blood_clot_path = None,
                  lite = False):
    """
    Break the slide into tiles of size tile_size using an overlap
    If eval is true calculate two measures of how much tissue there is in each tile
    
    Parameters
    ----------
    tile_dir : str
        The path to the tiles for a specific whole slide image.
    scene_metadata : dict
        The metadata dict must have the image resolution and the
        width and height in pixels.
    tile_size : int, optional
        The tile edge size in pixels. The default is 512.
    overlap : int, optional
        The overlap of the tiles in pixels. The default is 0.
    tissue_eval : Boolean, optional
        Binary flag for determining if each tile should be evaluated for tissue. 
        The default is True.
    find_nuclei : Boolean, optional
        Binary flag for determining if nuclei should be looked for in each slide. 
        The default is True.
    tissue_thresh : float, optional
        The threshold of tissue required for analysis. The default is 0.05.
    foreground_threshold : int, optional
        Threshold for separating foreground and background. The default is 140.
    min_nucleus_area : int, optional
        The minimum area of an object to be considered a nucleus (in square mcm). The default is 6.
    blood_clot_path : str, optional
        The path to the blood clot npz and json files. If not None, blood clot detections are 
        removed from consideration when detecting nuclei.
    lite : Boolean, optional
        Limits the procedure and outputs. The default is False.

    Returns
    -------
    tiles : list
        A list of dictionaries. One for each tile. 
        Each dictionary has these keysL items:
            tile: numpy array with the tile in it
            tile_size: 		size of the tile
            'overlap':		overlap 
            'zoom_level': 	zoom_level 
            'col':			what column the tile is in
            'row':			what row the tile is in.

    """
    if (find_nuclei):
        # For computational efficiency, need to evaluate presence of tissue prior to finding nuclei
        tissue_eval = True

    print(tile_dir)
    
    if blood_clot_path is not None:
        scene_name = tile_dir.split('/')[-1]
        blood_clot_mask, low_res_mag = read_annotation_info(blood_clot_path, scene_name)
        #  comment out this next line and replace with a tile generator for low memory operations
        blood_clot_mask = transform.resize(blood_clot_mask,
                                           (scene_metadata['sizeY'], scene_metadata['sizeX']),
                                           order = 0, preserve_range = True)
    
    tiles = []
    files = os.listdir(tile_dir)
    for idx, file in enumerate(files):
        print(str(idx) + ': ' + file)
        file_path = tile_dir + '/' + file
        tile = skimage.io.imread(file_path)
        
        fs = file.split('__')[1].split('_')
        row = int(fs[0][1:])
        col = int(fs[1][1:])
        tile_size = int(fs[2][2:])
        overlap = int(fs[3][2:].split('.')[0])
        
        if blood_clot_path is not None:
            #  replace blood clot detections with white pixels
            bcm_tile = blood_clot_mask[(row*tile_size):((row+1)*tile_size),
                                       (col*tile_size):((col+1)*tile_size)]
            
            bcm_idx = np.where(bcm_tile == 1)
            tile[(bcm_idx[0], bcm_idx[1])] = [255, 255, 255]
        
        this_tile = {
            'tile_size': tile_size,
            'overlap': overlap,
            'col': col,
            'row': row,
            'tissue': 0,
            'nuclei': [],
            'num_nuclei': 0
        }

        if (tissue_eval):
            tissue, tt = tileseg.find_tissue(tile)
            this_tile['tissue'] = tissue
            this_tile['nuclei'] = []
            this_tile['num_nuclei'] = 0

            if tile.shape != (tile_size, tile_size, 3):
                tissue = 0
                print('Tile has the incorrect shape')

            if (find_nuclei and tissue > tissue_thresh):
                try:
                    nuc = detect_nuclei(tile, scene_metadata['resolution'], 
                                        foreground_threshold = foreground_threshold, 
                                        min_nucleus_area = min_nucleus_area)
                    this_tile['nuclei'] = nuc
                    this_tile['num_nuclei'] = nuc['num_nuclei']
                except:
                    print('nuclei detection failed')
                if (lite):
                    del this_tile['nuclei']['im_nuclei_seg_mask']
                    del this_tile['nuclei']['objProps']

        if (lite):
            this_tile['tile'] = []
        else:
            this_tile['tile'] = tile

        tiles.append(this_tile)
    print('complete')
    return(tiles)

def read_annotation_info(annotation_omit_path, scene_name):
    annotation_mask = scipy.sparse.load_npz(f'{annotation_omit_path}/{scene_name}.npz').toarray()
    with open(f'{annotation_omit_path}/{scene_name}.json', 'r') as jsonfile:
        low_res_mag = json.load(jsonfile)['magnification']
    
    return(annotation_mask, low_res_mag)

def get_annotation_tile(annotation_mask, low_res_mag, tile_name, magnification = 20):
    # not in use yet
    # could be a way to parse the low res mag anno mask and keep low memory
    tile_params = tile_name.split('__')[1].split('_')
    row = int(tile_params[0][1:])
    col = int(tile_params[1][1:])
    tile_size = int(tile_params[2][2:])
    overlap = int(tile_params[3][2:].split('.')[0])
    
    mag_ratio = low_res_mag / magnification
    anno_chip_size_raw = mag_ratio * tile_size
    anno_chip_size = int(np.ceil(anno_chip_size_raw))
    anno_chip_size_iter = int(np.floor(anno_chip_size_raw))
    
    #anno_chip_low_res = annotation_mask[row*anno_chip_size_iter:(row+1)*anno_chip_size_iter
    #anno_chip = transform.resize()
    return
    
def find_fragments_from_file(path_to_image, low_res_mag, scene_id = 0, minimum_area = 1, valid_magnification = 20, intensity_threshold = 200, beta = 0.12):
    """
    Find and label the tissue fragments in a reduced resolution image.

    Parameters
    ----------
    path_to_image : str
        The path to the image to open.
    low_res_mag : int
        The magnification at which to detect the fragments. A good choice is 4.
    scene_id : int, optional
        The scene_id from the list of valid scenes. Valid scenes are determined
        by comparing the scene magnification to the valid_magnification 
        parameter. The default is 0.
    minimum_area : int, optional
        The minimum number of pixels a fragment must have to be labeled in the
        final mask. The default is 1.
    valid_magnification : int, optional
        The magnification of valid scenes in an image. The default is 20.
    intensity_threshold : int, optional
        Must be [0, 255] inclusive. The intensity threshold for assigning
        background pixels. The default is 200.
    beta : float, optional
        The optical density threshold for tissue detection. The default is 0.12.

    Returns
    -------
    Numpy array at the magnification specified by low_res_mag. The tissue 
    fragments are labeled with ascending integers.

    """    
    slide = image_loader.open_image(path_to_image, valid_magnification)
    slide.set_scene(scene_id)
    print(f'File Loaded: {path_to_image}')
    
    ## Low Resolution Image for visualization
    im_low_res = slide.get_low_res_image(low_res_mag)
    return(find_fragments(im_low_res, minimum_area = minimum_area, 
                          intensity_threshold = intensity_threshold, 
                          beta = beta))


def find_fragments(low_res_img_array, minimum_area = 1, intensity_threshold = 200, beta = 0.12):
    """
    Find and label the tissue fragments in a reduced resolution image.

    Parameters
    ----------
    low_res_img_array : numpy array
        The array of the low resolution image for which to label the tissue
        fragments.
    minimum_area : int, optional
        The minimum number of pixels a fragment must have to be labeled in the
        final mask. The default is 1.
    intensity_threshold : int, optional
        Must be [0, 255] inclusive. The intensity threshold for assigning
        background pixels. The default is 200.
    beta : float, optional
        The optical density threshold for tissue detection. The default is 0.12.

    Returns
    -------
    Numpy array at the magnification specified by low_res_mag. The tissue 
    fragments are labeled with ascending integers.

    """
    ## Low Resolution Image for visualization
    im_low_res = tileseg.standardize_image_background(low_res_img_array, intensity_threshold = intensity_threshold)
    
    # Convert to optical density values
    tile = tileseg.optical_density(im_low_res)
    
    # Threshold at beta and create binary image
    tile = np.max(tile, axis=2) >= beta
    
    # Remove small holes and islands in the image
    disk_radius = 1
    tile = morphology.binary_dilation(tile, morphology.disk(disk_radius))
    tile = ndimage.binary_fill_holes(tile)
    tile = morphology.binary_erosion(tile, morphology.disk(disk_radius))
    tissue_mask = morphology.remove_small_objects(tile, min_size = minimum_area)
    
    fragment_mask = measure.label(tissue_mask, connectivity=1)        
    return(fragment_mask)

def add_fragment_labels_to_nuclei(fragment_mask, nuclei_df, low_res_mag, original_magnification):
    """
    Adds a fragment_label column to the nuclei_df data frame object and returns
    the updated data frame.

    Parameters
    ----------
    fragment_mask : numpy array
        The labeled fragment mask output from find_fragments.
    nuclei_df : pandas data frame
        A data frame with nuclei position columns nuclei_x_wsi and nuclei_y_wsi.
    low_res_mag : int
        The magnification factor of fragment_mask.
    original_magnification : int
        The magnification factor of the original image which corresponds to the
        position information in nuclei_df.

    Returns
    -------
    The nuclei_df with an additional column: fragment_label.

    """
    mag_scale = low_res_mag / original_magnification
    low_res_x = (nuclei_df['nuclei_x_wsi'] * mag_scale).astype(int)
    low_res_y = (nuclei_df['nuclei_y_wsi'] * mag_scale).astype(int)
    frag_labels = fragment_mask[low_res_y, low_res_x]
    nuclei_df['fragment_label'] = np.where(frag_labels == 0, np.nan, frag_labels).astype(int)
    return(nuclei_df)
    
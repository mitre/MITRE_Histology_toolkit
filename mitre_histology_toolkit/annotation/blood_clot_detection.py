from ..nuclei_detection import tileseg
from skimage import measure, transform
from ..image import image_loader
from PIL import Image
import scipy.sparse
import numpy as np
import pickle
import json
import os

def get_image_info(image_path, image_name, low_res_mag = 4, magnification = 20):
    """
    

    Parameters
    ----------
    image_path : TYPE
        DESCRIPTION.
    image_name : TYPE
        DESCRIPTION.
    low_res_mag : TYPE, optional
        DESCRIPTION. The default is 4.
    magnification : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    None.

    """
    fpath = os.path.join(image_path, image_name)
    image_type = os.path.splitext(image_name)[1]
    main_tag = image_name.replace(image_type, '').replace(' ', '_').replace('.', '_')
    
    slide = image_loader.open_image(fpath, magnification)
    return(main_tag, slide)

def generate_tissue_mask(image_name, im_low_res, annotation_path = None, magnification = 20, low_res_mag = 4, scene_id = None):    
    """
    

    Parameters
    ----------
    image_name : TYPE
        DESCRIPTION.
    im_low_res : TYPE
        DESCRIPTION.
    annotation_path : TYPE, optional
        DESCRIPTION. The default is None.
    magnification : TYPE, optional
        DESCRIPTION. The default is 20.
    low_res_mag : TYPE, optional
        DESCRIPTION. The default is 4.
    scene_id : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    _, tissue_mask = tileseg.find_tissue(im_low_res)
    if annotation_path is not None:
        scipy_sparse_fname = f'{annotation_path}/{image_name}'.replace('.svs', '.npz')
        if os.path.isfile(scipy_sparse_fname):
            annotation_mask = scipy.sparse.load_npz(scipy_sparse_fname)
            annotation_mask = annotation_mask.toarray()
    
            with open(f'{annotation_path}/{image_name}'.replace('.svs', '.json'), 'r') as anno_file:
                annotation_params = json.loads(anno_file.read())
        
            # Remove tissue components
            for ii in annotation_params['not_synovium']:
                annotation_mask[annotation_mask == ii] = 0
    
            anno_mask = np.where(annotation_mask > 0, 1, 0)
            
            if tissue_mask.shape != anno_mask.shape:
                print(f'shape mismatch: {tissue_mask.shape}, and {anno_mask.shape}')
                anno_mask = transform.resize(anno_mask, tissue_mask.shape, order = 0, preserve_range = True).astype(int) # order = 0 uses nearest neighbor and retains binary mask
            
            tissue_mask = tissue_mask * anno_mask        

    return(tissue_mask)

def convert_minimum_area_threshold_to_px(minimum_area_threshold, low_res_mag, slide, scene_id = None):
    """
    Converts the minimum area threshold (given in square micrometers) to units 
    that can be compared with the low resolution image objects output by the 
    regionprops function.

    Parameters
    ----------
    minimum_area_threshold : TYPE
        DESCRIPTION.
    low_res_mag : TYPE
        DESCRIPTION.
    slide : TYPE
        DESCRIPTION.
    scene_id : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    scene_info = slide.get_info(scene_id = scene_id)
    mag_ratio = low_res_mag / scene_info['magnification']
    sq_mcm_to_px = scene_info['resolution'][0] * scene_info['resolution'][1]
    return((minimum_area_threshold / sq_mcm_to_px) * mag_ratio**2)
    
def predict_pixels(model_str, tissue_mask, im_low_res, blood_clot_model_path, minimum_area_threshold, low_res_mag, slide, scene_id = None):
    """
    

    Parameters
    ----------
    model_str : TYPE
        DESCRIPTION.
    tissue_mask : TYPE
        DESCRIPTION.
    im_low_res : TYPE
        DESCRIPTION.
    blood_clot_model_path : TYPE
        DESCRIPTION.
    minimum_area_threshold : TYPE
        DESCRIPTION.
    low_res_mag : TYPE
        DESCRIPTION.
    slide : TYPE
        DESCRIPTION.
    scene_id : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    if model_str == 'RGB':
        pass
    elif model_str == 'RGB_HSV':
        blood_mask = predict_pixels_rgb_hsv(tissue_mask, im_low_res, blood_clot_model_path)
    elif model_str == 'COMPOSITE':
        blood_mask = predict_pixels_rgb_hsv_composite(tissue_mask, im_low_res, blood_clot_model_path)

    blood_mask_label = measure.label(blood_mask, connectivity=1)
    rps = measure.regionprops(blood_mask_label)
    
    min_area_thresh_px = convert_minimum_area_threshold_to_px(minimum_area_threshold, 
                                                              low_res_mag, 
                                                              slide, 
                                                              scene_id = scene_id)
    coords_list = []
    for rp in rps:
        if rp.area > min_area_thresh_px:
            coords_list += [rp.coords]

    coords = np.concatenate(coords_list)
    
    blood_indices = (coords[:,0], coords[:,1])
    blood_mask_final = np.zeros(blood_mask.shape)
    blood_mask_final[blood_indices] = 1
    return(blood_mask_final)

def predict_pixels_rgb_hsv(tissue_mask, im_low_res, blood_clot_model_path):
    """
    

    Parameters
    ----------
    tissue_mask : TYPE
        DESCRIPTION.
    im_low_res : TYPE
        DESCRIPTION.
    blood_clot_model_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    with open(f'{blood_clot_model_path}/blood_clot_model_rgb_hsv.pkl', 'rb') as f:
        clf = pickle.load(f)

    tissue_indices = np.where(tissue_mask == 1)
    hsv_image = Image.fromarray(im_low_res.astype('uint8'))
    hsv_low_res = np.array(hsv_image.convert('HSV'))

    tissue_pixels = np.concatenate((im_low_res[tissue_indices], hsv_low_res[tissue_indices]), axis = 1)

    new_preds = clf.predict(tissue_pixels)
    new_pred_bin = (new_preds == 'Blood')

    blood_mask = np.zeros(tissue_mask.shape)
    blood_indices = (tissue_indices[0][new_pred_bin], tissue_indices[1][new_pred_bin])
    blood_mask[blood_indices] = 1
    return(blood_mask)

def predict_pixels_rgb_hsv_composite(tissue_mask, im_low_res, blood_clot_model_path):
    """
    

    Parameters
    ----------
    tissue_mask : TYPE
        DESCRIPTION.
    im_low_res : TYPE
        DESCRIPTION.
    blood_clot_model_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    with open(f'{blood_clot_model_path}/blood_clot_model_rgb_hsv_composite.pkl', 'rb') as f:
        clf = pickle.load(f)

    tissue_indices = np.where(tissue_mask == 1)
    hsv_image = Image.fromarray(im_low_res.astype('uint8'))
    hsv_low_res = np.array(hsv_image.convert('HSV'))

    tissue_pixels = np.concatenate((im_low_res[tissue_indices], hsv_low_res[tissue_indices]), axis = 1)

    new_preds = clf.predict(tissue_pixels)
    new_pred_bin = (new_preds == 'Blood')

    blood_mask = np.zeros(tissue_mask.shape)
    blood_indices = (tissue_indices[0][new_pred_bin], tissue_indices[1][new_pred_bin])
    blood_mask[blood_indices] = 1
    return(blood_mask)

def save_mask_and_metadata(blood_mask, blood_clot_metadata, blood_clot_path, main_tag):
    """
    

    Parameters
    ----------
    blood_mask : TYPE
        DESCRIPTION.
    blood_clot_metadata : TYPE
        DESCRIPTION.
    blood_clot_path : TYPE
        DESCRIPTION.
    main_tag : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sparse_matrix = scipy.sparse.csc_matrix(blood_mask)
    scipy.sparse.save_npz(f'{blood_clot_path}/{main_tag}.npz', sparse_matrix)
    
    with open(f'{blood_clot_path}/{main_tag}.json', "w") as outfile: 
        json.dump(blood_clot_metadata, outfile)

def run_blood_clot_detection(image_path, image_name, blood_clot_path, blood_clot_model_path, model_str = 'RGB_HSV', minimum_area_threshold = 3000, annotation_path = None, magnification = 20, low_res_mag = 4, scene_id = None):
    """
    

    Parameters
    ----------
    image_path : TYPE
        DESCRIPTION.
    image_name : TYPE
        DESCRIPTION.
    blood_clot_path : TYPE
        DESCRIPTION.
    blood_clot_model_path : TYPE
        DESCRIPTION.
    model_str : TYPE, optional
        DESCRIPTION. The default is 'RGB_HSV'.
    minimum_area_threshold : TYPE, optional
        DESCRIPTION. The default is 3000.
    annotation_path : TYPE, optional
        DESCRIPTION. The default is None.
    magnification : TYPE, optional
        DESCRIPTION. The default is 20.
    low_res_mag : TYPE, optional
        DESCRIPTION. The default is 4.
    scene_id : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    main_tag, slide = get_image_info(image_path, image_name, low_res_mag = low_res_mag, magnification = magnification)
    if scene_id is None:
        scene_id = slide.valid_scenes[0]
    
    im_low_res = slide.get_low_res_image(low_res_mag, scene_id = scene_id)
    scene_tag = f'{main_tag}_s{scene_id}'
    print(scene_tag)
    print(annotation_path)
    print(image_name)
    print(f'{annotation_path}/{image_name}'.replace('.svs', '.npz'))
    tissue_mask = generate_tissue_mask(image_name, im_low_res, annotation_path = annotation_path, magnification = magnification, low_res_mag = low_res_mag, scene_id = scene_id)
    blood_mask_final = predict_pixels(model_str, tissue_mask, im_low_res, blood_clot_model_path, minimum_area_threshold, low_res_mag, slide, scene_id = scene_id)
    blood_clot_metadata = {"magnification": low_res_mag, 
                           "minimum_area_sq_mcm": minimum_area_threshold}

    save_mask_and_metadata(blood_mask_final, blood_clot_metadata, blood_clot_path, scene_tag)

def run_blood_clot_detection_all_scenes(image_path, image_name, blood_clot_path, blood_clot_model_path, model_str = 'RGB_HSV', minimum_area_threshold = 3000, annotation_path = None, magnification = 20, low_res_mag = 4):
    main_tag, slide = get_image_info(image_path, image_name, low_res_mag = low_res_mag, magnification = magnification)
    for scene_id in slide.valid_scenes:
        run_blood_clot_detection(image_path, image_name, blood_clot_path, blood_clot_model_path, model_str = model_str, minimum_area_threshold = minimum_area_threshold, annotation_path = annotation_path, magnification = magnification, low_res_mag = low_res_mag, scene_id = scene_id)

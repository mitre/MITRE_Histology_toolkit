from ..image import image_loader
import scipy.sparse
import numpy as np
import skimage.io
import itertools
import skimage
import json
import os

def process_image(image_path, image_name, tile_path, annotation_path = None, magnification = 20, overlap = 0, tile_size = 1024):
    fpath = os.path.join(image_path, image_name)
    image_type = os.path.splitext(image_name)[1]
    main_tag = image_name.replace(image_type, '').replace(' ', '_').replace('.', '_')
    
    # load image and get scenes if necessary
    slide = image_loader.open_image(fpath, magnification)
    
    if annotation_path is not None:
        annotation_mask, annotation_params = get_annotation_objects(annotation_path, image_name, magnification)
    else:
        annotation_mask, annotation_params = (None, None)
        
    for scene_id in slide.valid_scenes:
        slide.set_scene(scene_id)
        scene_tag = f'{main_tag}_s{scene_id}'
        
        # Create tile folder
        tile_folder = os.path.join(tile_path, scene_tag)
        if not os.path.exists(tile_folder):
            os.makedirs(tile_folder)
        
        tile_iterator = get_tile_iterator(slide, magnification, tile_size, overlap)
        for tile_info in tile_iterator:
            print(tile_info)
            tile, row_index, col_index, keep_tile = get_tile_array(tile_info, slide, tile_size, magnification = magnification, annotation_mask = annotation_mask, annotation_params = annotation_params)
            
            if keep_tile:
                save_tile(tile, tile_folder, scene_tag, row_index, col_index, tile_size, overlap)
    
    print(f'Image {image_name} processed')
    if image_type == '.vsi':
        slide.close()
    
    return(0)

def get_annotation_objects(annotation_path, image_name, magnification):
    # Load annotation
    annotation_param_path = f'{annotation_path}/{image_name}'.replace('.svs', '.json')
    annotation_mask_path = f'{annotation_path}/{image_name}'.replace('.svs', '.npz')
    with open(annotation_param_path, 'r') as anno_file:
        annotation_params = json.loads(anno_file.read())
    
    annotation_mask = scipy.sparse.load_npz(annotation_mask_path)
    annotation_mask = annotation_mask.toarray()

    # Remove tissue components
    for ii in annotation_params['not_synovium']:
        annotation_mask[annotation_mask == ii] = 0
    
    return(annotation_mask, annotation_params)

def get_tile_iterator(scene, magnification, tile_size, overlap):
    scene_info = scene.get_info()
    scene_dim = [scene_info['sizeX'], scene_info['sizeY']]
    tile_iterator = itertools.product(range(scene_dim[0] // tile_size),
                                      range(scene_dim[1] // tile_size))
    return(tile_iterator)

def get_tile_array(tile_info, scene, tile_size, magnification, annotation_mask = None, annotation_params = None):
    col_index, row_index = tile_info
    col_start = tile_size * col_index
    row_start = tile_size * row_index
    # gets numpy array of the region between the boundary (at the original magnification) scaled to `mag`
    tile = scene.get_region(col_start, col_start + tile_size, row_start, 
                            row_start + tile_size, magnification)
    
    # TODO screen tiles and only keep ones with tissue
    keep_tile = True
    if annotation_mask is not None:
        col_start = tile_size * col_index
        row_start = tile_size * row_index
        col_end = tile_size * (col_index + 1)
        row_end = tile_size * (row_index + 1)
        
        keep_tile = False
        mag_adjust = magnification // annotation_params['magnification']
        mm = annotation_mask[
            row_start//mag_adjust : row_end//mag_adjust,
            col_start//mag_adjust : col_end//mag_adjust, 
            ]
        if np.sum(mm) > 0:
            keep_tile = True
    
    return(tile, row_index, col_index, keep_tile)

def save_tile(image, tile_folder, scene_tag, row_index, col_index, tile_size, overlap):
    save_name = f'{scene_tag}__R{row_index}_C{col_index}_TS{tile_size}_OL{overlap}.jpg'
    save_path = os.path.join(tile_folder, save_name)
    skimage.io.imsave(save_path, image, quality = 100)
    return

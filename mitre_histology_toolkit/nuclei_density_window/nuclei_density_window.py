from scipy import signal, sparse
import cv2
import numpy as np
import os
import pandas as pd
import skimage.io
## Loading custom scripts
from mitre_histology_toolkit.nuclei_detection import tileseg

# %%===========================================================================
# =============================================================================

def tiles_to_img(tile_dir, mask_dir,  resize_factor=4):
    '''
    Takes a directory of tiles and resizes it and stores it as a dict
    
    Arguments
    -------------
    tile_dir : str
        directory of tile images
    mask_dir : str
        directory of tissue masks
    scene_name : str
        name of the scene
    resize_factor : int
        factor to resize the tiles by
    
    Returns
    -------------
    stack : dict
        dict with tile (row,col) coordinates as keys and tile image and mask as values
    '''
    
    
    stack = {}
    scene_name = os.path.basename(os.path.normpath(tile_dir))

    for file in os.listdir(tile_dir):
        params = file.split('__')[1].split('_')
        row = int(params[0][1:])
        col = int(params[1][1:])
        
        #Read image
        im = skimage.io.imread(tile_dir + file)
        im = cv2.resize(im, (im.shape[0] // resize_factor, im.shape[1] // resize_factor))
        

        #Read mask
        mask_fname = mask_dir + f'{scene_name}__R{row}_C{col}_TS1024_OL0_nuclei_mask.npz'

        if os.path.isfile(mask_fname):
            sparse_matrix = sparse.load_npz(mask_fname)
            mask = sparse_matrix.todense().astype('int')
            mask = cv2.resize(mask, (im.shape[0], im.shape[1]), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros([im.shape[0], im.shape[1]])
        stack[(row,col)] = (im, mask.astype(bool))
    
    return stack

# %%===========================================================================
# =============================================================================

def grab_region(stack, rows = None, cols = None):
    '''
    Takes a specified region of tiles and creates the image and mask
    
    Arguments
    -------------
    stack : dict
        dict with tile (row,col) coordinates as keys and tile image and mask as values
    rows : tuple
        Tuple with min row and max row of region of interest
    cols : tuple
        Tuple with min col and max col of region of interest
    
    Returns
    -------------
    full_image : ndarray
        Image of region of interest
    full_mask : ndarray
        Mask of region of interest
    tissue_mask : ndarray
        image of region of interest after its background has been standardized and converted to optical densities
    '''
    if rows is None:
        rows = [k[0] for k in stack.keys()]
        rows = (min(rows), max(rows))
    if cols is None:
        cols = [k[1] for k in stack.keys()]
        cols = (min(cols), max(cols))
        
    im_init = True
    new_col = True
    im_shape = list(stack.items())[0][1][0].shape

    for row in range (rows[0],rows[1]):
        new_row = True
        for col in range(cols[0], cols[1]):
            if (row,col) in stack:
                im, mask = stack[(row,col)]

            else: 

                im = np.ones(im_shape).astype(int) * 255
                mask = np.zeros([im_shape[0], im_shape[1]]).astype(int)
            
            if new_row:
                new_row = False
                row_image = im
                row_mask = mask
            else: 
                row_image = np.concatenate((row_image, im), 1)
                row_mask = np.concatenate((row_mask, mask), 1)
                
        if new_col:
            new_col = False
            full_image = row_image
            full_mask = row_mask
        else:
            full_image = np.concatenate((full_image, row_image), 0)
            full_mask = np.concatenate((full_mask, row_mask), 0)
    
    tissue_amount , tissue_mask = tileseg.find_tissue(full_image)
    
    return full_image, full_mask, tissue_mask

# %%===========================================================================
# =============================================================================

def construct_local_regional_window(full_mask, tissue_mask,tissue_thresh = .25, ws =64, wb=512):
    '''
    Calculates the local/regional scores 
    
    Arguments
    -------------
    full_mask : ndarray
        Mask of region of interest
    tissue_mask : ndarray
        image of region of interest after its background has been standardized and converted to optical densities
    tissue_thresh: number in (0,1)
        threshold for tissue detection
    ws : int
        convolution filter for local window
    wb : int
        convolution filter for regional window
    
    Returns
    -------------
    small : ndarray
        local window
    big : ndarray
        regional window
    tissue_small : ndarray 
        fft local window
    rel : ndarray
        local/regional window
    '''
    small = signal.fftconvolve(full_mask, np.ones([ws,ws]), mode = 'same')
    tissue_small = signal.fftconvolve(tissue_mask, np.ones([ws,ws])/(ws**2), mode = 'same')
    tissue_small[tissue_small<tissue_thresh] = -1 
    small = small/ (tissue_small * (0.5e-6) * (ws**2))
    
    big = signal.fftconvolve(full_mask, np.ones([wb,wb]), mode = 'same')
    tissue_big = signal.fftconvolve(tissue_mask, np.ones([wb,wb])/(wb**2), mode = 'same')
    tissue_big[tissue_big<tissue_thresh] = -1
    big = big / (tissue_big * (0.5e-6) * (wb**2))
    
    rel = small/big
    rel[tissue_small < 0] = 0
    rel[rel<0] = 0
    
    return small, big, tissue_small, rel
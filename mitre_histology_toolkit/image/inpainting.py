import cv2
import skimage
import numpy as np
from mitre_histology_toolkit.nuclei_detection import tileseg
def grab_region_fill(stack, rows = None, cols = None):
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
    full_dst : ndarry
        image of region of interest with 
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
                # print(row,col)
                im, mask = stack[(row,col)]
                aug_mask = skimage.morphology.dilation(mask, skimage.morphology.disk(1))
                dst = cv2.inpaint(im.astype(np.uint8),aug_mask.astype(np.uint8),10,cv2.INPAINT_TELEA)

            else: 
                im = np.ones(im_shape).astype(int) * 255
                mask = np.zeros([im_shape[0], im_shape[1]]).astype(int)
                dst = np.zeros(im_shape).astype(int) * 255
            
            if new_row:
                new_row = False
                row_image = im
                row_mask = mask
                row_dst = dst
            else: 
                row_image = np.concatenate((row_image, im), 1)
                row_mask = np.concatenate((row_mask, mask), 1)
                row_dst = np.concatenate((row_dst,dst),1)
                
        if new_col:
            # print(row)
            new_col = False
            full_image = row_image
            full_mask = row_mask
            full_dst = row_dst
        else:
            full_image = np.concatenate((full_image, row_image), 0)
            full_mask = np.concatenate((full_mask, row_mask), 0)
            full_dst = np.concatenate((full_dst, row_dst), 0)

    tissue_amount , tissue_mask = tileseg.find_tissue(full_image)
    
    return full_image, full_mask, tissue_mask,full_dst

def pad_nuc_fill(filled, shape_tuple, rescale):
    lx = round(shape_tuple[1]/rescale)-filled.shape[0]
    ly = round(shape_tuple[0]/rescale)-filled.shape[1]
    channels = []
    for i in range(3):
        channels.append(np.pad(filled[:,:,i], ((lx,0),(ly,0)), 'constant',constant_values = 255))
    return np.stack(channels,axis = 2)


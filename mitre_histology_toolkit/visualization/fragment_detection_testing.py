from mitre_histology_toolkit.nuclei_detection import slide_to_tiles as stt
from scipy.ndimage import binary_fill_holes
from skimage import morphology, measure
import matplotlib.pyplot as plt
import numpy as np
import os

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

## Read slide
os.chdir('C:/Users/dfrezza/Documents/SpyderWorkspace/ai-histology')
slide_dir = 'data/raw/slides/AMP/'

# for filename in os.listdir(slide_dir):
#     fpath = f'{slide_dir}/{filename}'
#     if os.path.isfile(fpath):
# print(filename)

filename = '300-142.vsi'
image_type = os.path.splitext(filename)[1]
fpath = f'{slide_dir}/{filename}'
valid_magnification = 20
slide, valid_scenes = stt.open_image(fpath, image_type, valid_magnification)
print(f'File Loaded: {filename}')

## Low Resolution Image for visualization
if image_type == '.vsi':
    slide.idx = valid_scenes[0]
    scene = slide
else:
    scene = slide.get_scene(valid_scenes[0])

low_res_mag = 4
im_low_res = stt.get_block_from_image(image_type, scene, low_res_mag)

disk_radius = 1

im2 = im_low_res.copy()
im2[np.where(im2.min(axis = 2) > 200)] = (255, 255, 255)

plot_dir = 'mitre_histology_toolkit/visualization/fragments/'
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].imshow(im_low_res.min(axis = 2))
ax[1].imshow(im2.min(axis = 2))
# fig.savefig(f'{plot_dir}/initial_{filename.split(".")[0]}.png')
# plt.close('all')

# Convert to optical density values
tile = optical_density(im2)
# Threshold at beta and create binary image
beta = 0.12
tile = np.max(tile, axis=2) >= beta
# Remove small holes and islands in the image
tile = morphology.binary_dilation(tile, morphology.disk(disk_radius))
plt.imshow(tile)
tile = binary_fill_holes(tile)
tile = morphology.binary_erosion(tile, morphology.disk(disk_radius))
minimum_area = 100000
tissue_mask = morphology.remove_small_objects(tile, min_size=minimum_area)

# Calculate percentage of tile containig tissue
percentage = np.mean(tissue_mask)
tissue_amount = percentage #>= tissue_threshold

im_label = measure.label(tissue_mask, connectivity=1)        
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].imshow(im_low_res)
axs[1].imshow(tissue_mask)
# fig.savefig(f'{plot_dir}/final_{filename.split(".")[0]}.png')
# plt.close('all')
        
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].imshow(im_low_res)
axs[1].imshow(im_label)

regions = measure.regionprops(im_label)
for rg in regions:
    print(rg.area)

###############################################################################
###############################################################################
###############################################################################

## Compare tissue deteection methods

from mitre_histology_toolkit.nuclei_detection import slide_to_tiles as stt
from scipy.ndimage import binary_fill_holes
from skimage import morphology, measure
import matplotlib.pyplot as plt
import numpy as np
import os

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

def old_tissue_alg(low_res_mag, image_type, scene, minimum_area, beta):
    im_low_res = stt.get_block_from_image(image_type, scene, low_res_mag)
    
    disk_radius = 1
        
    # Convert to optical density values
    tile = optical_density(im_low_res)
    # Threshold at beta and create binary image
    tile = np.min(tile, axis=2) >= beta
    # Remove small holes and islands in the image
    tile = morphology.binary_dilation(tile, morphology.disk(disk_radius))
    tile = binary_fill_holes(tile)
    tile = morphology.binary_erosion(tile, morphology.disk(disk_radius))
    
    tissue_mask = morphology.remove_small_objects(tile, min_size=minimum_area)
    return(tissue_mask)

def new_tissue_alg(low_res_mag, image_type, scene, minimum_area, beta):
    im_low_res = stt.get_block_from_image(image_type, scene, low_res_mag)
    
    disk_radius = 1
    
    im2 = im_low_res.copy()
    im2[np.where(im2.min(axis = 2) > 200)] = (255, 255, 255)
    
    # Convert to optical density values
    tile = optical_density(im2)
    # Threshold at beta and create binary image
    tile = np.max(tile, axis=2) >= beta
    # Remove small holes and islands in the image
    tile = morphology.binary_dilation(tile, morphology.disk(disk_radius))
    tile = binary_fill_holes(tile)
    tile = morphology.binary_erosion(tile, morphology.disk(disk_radius))
    
    tissue_mask = morphology.remove_small_objects(tile, min_size=minimum_area)
    return(tissue_mask)

## Read slide
os.chdir('C:/Users/dfrezza/Documents/SpyderWorkspace/ai-histology')
slide_dir = 'data/raw/slides/HSS_RA/'

ii0 = 0
for filename in os.listdir(slide_dir):
    fpath = f'{slide_dir}/{filename}'
    if os.path.isfile(fpath):
        print(filename)
        image_type = os.path.splitext(filename)[1]
        fpath = f'{slide_dir}/{filename}'
        valid_magnification = 20
        slide, valid_scenes = stt.open_image(fpath, image_type, valid_magnification)
        print(f'File Loaded: {filename}')
        
        ## Low Resolution Image for visualization
        if image_type == '.vsi':
            slide.idx = valid_scenes[0]
            scene = slide
        else:
            scene = slide.get_scene(valid_scenes[0])
        
        low_res_mag = 4
        minimum_area = 100000
        beta = 0.12
        old_tile = old_tissue_alg(low_res_mag, image_type, scene, minimum_area, beta)
        new_tile = new_tissue_alg(low_res_mag, image_type, scene, minimum_area, beta)
        
        plot_dir = 'mitre_histology_toolkit/visualization/fragments/comparison'
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].imshow(old_tile)
        ax[1].imshow(new_tile)
        perc_tiss = (old_tile * new_tile).sum() / old_tile.sum()
        fig.suptitle(f'Percent of Tissue in Common: {perc_tiss:.3f}')
        fig.savefig(f'{plot_dir}/{filename.split(".")[0]}.png')
        plt.close('all')
        
        ii0 += 1
        if ii0 > 20:
            break


###############################################################################
###############################################################################
###############################################################################

## Compare tissue deteection method to HTK
##  not implemented. old method mirrors htk method 

from mitre_histology_toolkit.nuclei_detection import slide_to_tiles as stt
from scipy.ndimage import binary_fill_holes
from skimage import morphology, measure
import matplotlib.pyplot as plt
import numpy as np
import os

import girder_client
import numpy as np
from matplotlib import pylab as plt
from matplotlib.colors import ListedColormap
from histomicstk.saliency.tissue_detection import get_slide_thumbnail, get_tissue_mask

APIURL = 'http://candygram.neurology.emory.edu:8080/api/v1/'
# SAMPLE_SLIDE_ID = '5d586d57bd4404c6b1f28640'
SAMPLE_SLIDE_ID = "5d817f5abd4404c6b1f744bb"

gc = girder_client.GirderClient(apiUrl=APIURL)
# gc.authenticate(interactive=True)
_ = gc.authenticate(apiKey='kri19nTIGOkWH01TbzRqfohaaDWb6kPecRqGmemb')


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

def old_tissue_alg(low_res_mag, image_type, scene, minimum_area, beta):
    im_low_res = stt.get_block_from_image(image_type, scene, low_res_mag)
    
    disk_radius = 1
        
    # Convert to optical density values
    tile = optical_density(im_low_res)
    # Threshold at beta and create binary image
    tile = np.min(tile, axis=2) >= beta
    # Remove small holes and islands in the image
    tile = morphology.binary_dilation(tile, morphology.disk(disk_radius))
    tile = binary_fill_holes(tile)
    tile = morphology.binary_erosion(tile, morphology.disk(disk_radius))
    
    tissue_mask = morphology.remove_small_objects(tile, min_size=minimum_area)
    return(tissue_mask)

def new_tissue_alg(low_res_mag, image_type, scene, minimum_area, beta):
    im_low_res = stt.get_block_from_image(image_type, scene, low_res_mag)
    
    disk_radius = 1
    
    im2 = im_low_res.copy()
    im2[np.where(im2.min(axis = 2) > 200)] = (255, 255, 255)
    
    # Convert to optical density values
    tile = optical_density(im2)
    # Threshold at beta and create binary image
    tile = np.max(tile, axis=2) >= beta
    # Remove small holes and islands in the image
    tile = morphology.binary_dilation(tile, morphology.disk(disk_radius))
    tile = binary_fill_holes(tile)
    tile = morphology.binary_erosion(tile, morphology.disk(disk_radius))
    
    tissue_mask = morphology.remove_small_objects(tile, min_size=minimum_area)
    return(tissue_mask)

## Read slide
os.chdir('C:/Users/dfrezza/Documents/SpyderWorkspace/ai-histology')
slide_dir = 'data/raw/slides/HSS_RA/'

ii0 = 0
for filename in os.listdir(slide_dir):
    fpath = f'{slide_dir}/{filename}'
    if os.path.isfile(fpath):
        print(filename)
        image_type = os.path.splitext(filename)[1]
        fpath = f'{slide_dir}/{filename}'
        valid_magnification = 20
        slide, valid_scenes = stt.open_image(fpath, image_type, valid_magnification)
        print(f'File Loaded: {filename}')
        
        ## Low Resolution Image for visualization
        if image_type == '.vsi':
            slide.idx = valid_scenes[0]
            scene = slide
        else:
            scene = slide.get_scene(valid_scenes[0])
        
        low_res_mag = 4
        minimum_area = 100000
        beta = 0.12
        old_tile = old_tissue_alg(low_res_mag, image_type, scene, minimum_area, beta)
        new_tile = new_tissue_alg(low_res_mag, image_type, scene, minimum_area, beta)
        
        plot_dir = 'mitre_histology_toolkit/visualization/fragments/comparison'
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].imshow(old_tile)
        ax[1].imshow(new_tile)
        perc_tiss = (old_tile * new_tile).sum() / old_tile.sum()
        fig.suptitle(f'Percent of Tissue in Common: {perc_tiss:.3f}')
        fig.savefig(f'{plot_dir}/{filename.split(".")[0]}.png')
        plt.close('all')
        
        ii0 += 1
        if ii0 > 20:
            break
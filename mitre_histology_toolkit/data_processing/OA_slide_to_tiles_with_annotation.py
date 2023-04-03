#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:22:58 2021

@author: sguan
"""
import os
import numpy as np
import pandas as pd
import skimage
import skimage.io
from skimage.measure import label, regionprops
import large_image
import openslide
import matplotlib.pyplot as plt
from sys import argv
import json
import scipy.sparse
import cv2

def process_svs(file, fpath, annotation_path, output_path, over_lap=1):
    main_tag = file.replace('.svs', '')
    main_tag = main_tag.replace(' ', '_')
    ts = large_image.getTileSource(fpath)
    
    # Create tile folder
    tile_folder = output_path + main_tag
    if not os.path.exists(tile_folder):
        os.makedirs(tile_folder)
    
    # Load annotation
    anno_fpath = annotation_path + file.replace('.svs', '.json')
    mask_fpath = annotation_path + file.replace('.svs', '.npz')
    f = open(anno_fpath)
    anno = json.loads(f.read())
    mask = scipy.sparse.load_npz(mask_fpath)
    mask = mask.toarray()

    # Remove tissue components
    for ii in anno['not_synovium']:
        mask[mask==ii] = 0
    
    # Calculate allowable bounding box overlap
    ol = np.floor(over_lap * 1024)
    
    # Calculate region properties
    props = regionprops(mask)
    
    bbox = []
    magnification = 20
    for pp in props:
        bb = np.asarray(pp.bbox) * magnification/anno['magnification']
        bb = bb - np.asarray([ol, ol, -ol, -ol])
        bbox.append(bb)
    
    tile_iterator = ts.tileIterator(
        scale=dict(magnification=magnification),
        tile_size=dict(width=1024, height=1024),
        tile_overlap=dict(x=0, y=0),
        format=large_image.tilesource.TILE_FORMAT_NUMPY)
    

    
    for tile_info in tile_iterator:
        
        col = tile_info['level_x']
        row = tile_info['level_y']
        
        col_wsi_start = col * tile_info['width']
        row_wsi_start = row * tile_info['height']
        col_wsi_end = (col+1) * tile_info['width']
        row_wsi_end = (row+1) * tile_info['height']        
        
        keep = False
        
        for bb in bbox:
            if (row_wsi_start > bb[0] and
                row_wsi_end < bb[2] and
                col_wsi_start > bb[1] and
                col_wsi_end < bb[3]):
                keep = True
               
                
        if (keep):
            tile = np.array(tile_info['tile'])[:,:,:3]
            plt.imshow(tile)
            
            save_name = main_tag + '__R' + str(row) + '_C'+ str(col) + '_TS' + str(tile_size) + '_OL' + str(overlap)+'.jpg'
            output_path = tile_folder + '/' + save_name
            skimage.io.imsave(output_path, tile, quality=100)
            
def stringFilter(list1, list2):
    return [n for n in list1 if
             any(m in n for m in list2)]


## Define slide location and tile parameters
#file = argv[1]
#output_path = './tiles/'
#file_path = '../OA_slides/'
file = '004_SYN_207643.svs'
output_path = '../../data/processed/OA_tiles/'
file_path = '../../data/raw/OA/'
annotation_path = '../../data/processed/OA_annotations/'
lowres_path = '../../data/processed/OA_low_res/'

tile_size=1024
overlap=0
tissue_thresh=0.01

## Perform tiling
fpath = file_path + file
file_type = file[-3:]

try:
    process_svs(file, fpath, annotation_path, output_path)        
except Exception as e:
    print('error: ' + file)
    print(e)

    

## Create image thumbnail
main_tag = file.replace('.svs', '')
main_tag = main_tag.replace(' ', '_')
tile_folder = output_path + main_tag

images = os.listdir(tile_folder)
images = stringFilter(images, ['.jpg'])

for ii, fname in enumerate(images):
    params = fname.split('__')[1].split('_')
    
    row = int(params[0][1:])
    col = int(params[1][1:])
    
    if ii == 0:
        min_row = row
        max_row = row
        min_col = col
        max_col = col
    
    if (row < min_row):
        min_row = row
    
    if (row > max_row):
        max_row = row
        
    if (col < min_col):
        min_col = col
    
    if (col > max_col):
        max_col = col
    

for row in range(min_row, max_row):
    for col in range(min_col, max_col):
        fname = main_tag + '__R' + str(row) + '_C'+ str(col) + '_TS' + str(tile_size) + '_OL' + str(overlap)+'.jpg'
        fpath = tile_folder + '/' + fname
        try:
            im_orig = skimage.io.imread(fpath)
            im = cv2.resize(im_orig, (256,256))
        except:
            im = np.zeros([256,256,3])
        if col==min_col:
            image_col = im
        else:
            image_col = np.concatenate([image_col, im], axis=1)
            
    if row==min_row:
        image = image_col
    else:
        image = np.concatenate([image, image_col], axis=0)
    

# Create thumbnail folder
if not os.path.exists(lowres_path):
    os.makedirs(lowres_path)

output_path = lowres_path + main_tag + '.jpg'

skimage.io.imsave(output_path, image, quality=100)



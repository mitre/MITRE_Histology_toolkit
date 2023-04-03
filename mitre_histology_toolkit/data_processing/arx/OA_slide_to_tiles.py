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
import large_image
import openslide
import matplotlib.pyplot as plt
from sys import argv


def process_svs(fpath, output_path):
    main_tag = file.replace('.svs', '')
    main_tag = main_tag.replace(' ', '_')
    ts = large_image.getTileSource(fpath)
    
    # Create tile folder
    tile_folder = output_path + main_tag
    if not os.path.exists(tile_folder):
        os.makedirs(tile_folder)
    
    
    tile_iterator = ts.tileIterator(
        scale=dict(magnification=20),
        tile_size=dict(width=1024, height=1024),
        tile_overlap=dict(x=0, y=0),
        format=large_image.tilesource.TILE_FORMAT_NUMPY)
    
    for tile_info in tile_iterator:
        
        col = tile_info['level_x']
        row = tile_info['level_y']
        
        
        tile = np.array(tile_info['tile'])[:,:,:3]
        plt.imshow(tile)
        
        save_name = main_tag + '__R' + str(row) + '_C'+ str(col) + '_TS' + str(tile_size) + '_OL' + str(overlap)+'.jpg'
        output_path = tile_folder + '/' + save_name
        skimage.io.imsave(output_path, tile, quality=100)
            
def stringFilter(list1, list2):
    return [n for n in list1 if
             any(m in n for m in list2)]


## Define slide location and tile parameters
file = argv[1]
output_path = './tiles/'
file_path = '../OA_slides/'
tile_size=1024
overlap=0
tissue_thresh=0.01

## Perform tiling
fpath = file_path + file
file_type = file[-3:]

try:
    process_svs(fpath, output_path)        
except:
    print('error: ' + file)
    

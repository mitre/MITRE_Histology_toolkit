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
import slideio
from read_vsi import read_vsi

output_path = '../../data/processed/belgium_tiles/'
file_path = '../../data/raw/Belgium/'
tile_size=1024
overlap=0
tissue_thresh=0.01



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
        

def process_czi(fpath, output_path):
    slide = slideio.open_slide(fpath, "CZI")
    main_tag = file.replace('.czi', '')
    main_tag = main_tag.replace(' ', '_')
    for ii in range(0, slide.num_scenes):
        scene = slide.get_scene(ii)
        scene_dim = scene.size
        scene_tag = main_tag + '_s' + str(ii)
        # Create tile folder
        tile_folder = output_path + scene_tag
        if not os.path.exists(tile_folder):
            os.makedirs(tile_folder)
        
        for yi in range(0, scene_dim[0]//tile_size):
            for xi in range(0, scene_dim[1]//tile_size):
                ys = tile_size * yi
                xs = tile_size * xi
                image = scene.read_block((ys,xs,tile_size,tile_size))
                image = image[:,:,::-1]
                save_name = scene_tag+ '__R' + str(xi) + '_C'+ str(yi) + '_TS' + str(tile_size) + '_OL' + str(overlap)+'.jpg'
                save_path = tile_folder + '/' + save_name
                skimage.io.imsave(save_path, image, quality=100)
        
    
def stringFilter(list1, list2):
    return [n for n in list1 if
             any(m in n for m in list2)]

files = os.listdir(file_path)
files = stringFilter(files, ['.scn'])



for file in files:
    fpath = file_path + file
    file_type = file[-3:]
    
    try:
        if file_type == 'czi':
            process_czi(fpath, output_path)
            
        if file_type == 'svs':
            process_svs(fpath, output_path)
            
    except:
        print('error: ' + file)
        continue



skip_scenes=2




slide = slideio.open_slide(fpath, "SCN")
main_tag = file.replace('.scn', '')
main_tag = main_tag.replace(' ', '_')
metadata = []

for ii in range(0, slide.num_scenes):
    scene = slide.get_scene(ii)
    meta = {
        'idx': ii,
        'rect': scene.rect,
        'mag':scene.magnification
        }
    metadata.append(meta)
    
    
    # # Assume we want 20X magnifications
    # scale = scene.magnification / 20
    # scaled_tile_size = int(np.round(tile_size*scale))
    # scene_dim = scene.size
    # scene_tag = main_tag + '_s' + str(ii)
    # # Create tile folder
    # tile_folder = output_path + scene_tag
    # if not os.path.exists(tile_folder):
    #     os.makedirs(tile_folder)
    
    # for yi in range(0, scene_dim[0]//tile_size):
    #     for xi in range(0, scene_dim[1]//tile_size):
    #         ys = tile_size * yi
    #         xs = tile_size * xi
    #         image = scene.read_block((ys,xs,scaled_tile_size,scaled_tile_size),
    #                                  (tile_size,tile_size))
    #         save_name = scene_tag+ '__R' + str(xi) + '_C'+ str(yi) + '_TS' + str(tile_size) + '_OL' + str(overlap)+'.jpg'
    #         save_path = tile_folder + '/' + save_name
    #         skimage.io.imsave(save_path, image, quality=100)

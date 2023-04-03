import sys
sys.path.insert(1, 'src/cellularity_segmentation/')
import os
os.environ['JAVA_HOME'] = "C:/Program Files/Java/jdk1.8.0_301"
import image_loader 
import javabridge
import bioformats as bf
javabridge.start_vm(class_path=bf.JARS, max_heap_size='8G', run_headless=True)

import os
import numpy as np
import pandas as pd
import skimage
import skimage.io
import large_image
import openslide
import slideio

output_path = 'data/processed/AMP_tiles/'
file_path = 'data/raw/AMP/'

tile_size=1024
overlap=0
tissue_thresh=0.01

def process_vsi(fpath, output_path):
    slide = image_loader.ImageLoaderBioformat(fpath,manage_javabridge=False)

    # to get a specific region of a scene
    slide.get_valid_index() # gets a set of valid scene indices
    slide.idx = 0 # set the scene requested
    
    
    
    main_tag = file.replace('.vsi', '')
    main_tag = main_tag.replace(' ', '_')
    for index in slide.get_valid_index():
        slide.idx = index
        slide_info = slide.get_info()
        mag = int(slide_info['magnification'])
        if int(mag) != 20:
            continue
        
        scene_dim = [slide_info['sizeY'], slide_info['sizeX']]
        scene_tag = main_tag + '_s' + str(index)
        # Create tile folder
        tile_folder = output_path + scene_tag
        if not os.path.exists(tile_folder):
            os.makedirs(tile_folder)
        
        # to get a specific region of a scene

        
        
        for yi in range(0, scene_dim[0]//tile_size):
            for xi in range(0, scene_dim[1]//tile_size):
                ys = tile_size * yi
                xs = tile_size * xi
                image = slide.get_region(xs, xs + tile_size, ys, ys+tile_size, mag) # gets nparray of the region between the boundary (at the original magnification) scaled to `mag`

                save_name = scene_tag+ '__R' + str(yi) + '_C'+ str(xi) + '_TS' + str(tile_size) + '_OL' + str(overlap)+'.jpg'
                save_path = tile_folder + '/' + save_name
                skimage.io.imsave(save_path, image, quality=100)
    
def stringFilter(list1, list2):
    return [n for n in list1 if
             any(m in n for m in list2)]

files = os.listdir(file_path)
files = stringFilter(files, ['.vsi'])

import warnings

for file in files:
    print(file)
    fpath = file_path + file
    file_type = file[-3:]
    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        process_vsi(fpath, output_path)
    sys.stdout = save_stdout

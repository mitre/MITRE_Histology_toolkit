import os
import sys
os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home"

import javabridge
import bioformats as bf
import bioformats.omexml as ome
import numpy as np
from bs4 import BeautifulSoup
import re

res_threashold = 1
def read_vsi(filepath):
    # start java VM
    javabridge.start_vm(class_path=bf.JARS, max_heap_size='8G', run_headless=True)

    # split file names  
    basename = os.path.basename(filepath)
    print(basename)
    filename_root = os.path.splitext(os.path.basename(filepath))[0]
    

    # parse metadata
    md = bf.get_omexml_metadata(filepath)
    soup = BeautifulSoup(md, 'xml')
    objectives = {}
    for o in soup.select("Instrument > Objective"):
        objectives[o['ID']] = float(o['NominalMagnification'])
    
    images = []
    output = []
    size_tags = ['Size' + c for c in ['T', 'Z', 'Y', 'X', 'C']]
    res_tags = ['PhysicalSize' + c for c in ['Y', 'X']]
    for idx, image in enumerate(soup.select("Image")):
        try:
            # only keep images with objectivesettings (magnification), pixel resolution and pixel physical size
            objective = image.find("ObjectiveSettings")
            magnification = objectives[objective['ID']]
            pixels = image.find('Pixels')
            
            res = tuple([float(pixels[t]) for t in res_tags])
            sizes= tuple([int(pixels[t]) for t in size_tags])
            images.append({
                "pixel_size": res,
                "resolution": (sizes[2], sizes[3]),
                "magnification": magnification,
                "name": image['Name'],
                "idx": idx
            })
        except:
            print(f"skipping image {idx} because objectivesettings (magnification), pixel resolution, or pixel physical size is missing")
            pass


    for image in images:
        print(f"{image['idx']}: {image['name']} {str(image['resolution'])}, pixel size: {str(image['pixel_size'])}")

    # get image
    for i in images:
        if i['resolution'][0] * i['resolution'][1] < (2**31 - 1) and i['pixel_size'][0] < res_threashold:
            print(f"loading image {i['idx']}")
            output.append({
                "image_data": bf.load_image(filepath, series=i['idx'], rescale=False),
                "image_metadata": i
                })
        else:
            print(f"skip image {i['idx']} because either it's too large for java array ({i['resolution'][0]} x {i['resolution'][1]}) or pixel physical size ({i['pixel_size'][0]} um/px) above threshold")


    javabridge.kill_vm()
    return output


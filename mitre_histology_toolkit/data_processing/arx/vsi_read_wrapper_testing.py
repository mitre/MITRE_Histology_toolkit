import sys
sys.path.insert(1, 'src/cellularity_segmentation/')
import os
os.environ['JAVA_HOME'] = "C:/Program Files/Java/jdk1.8.0_301"
import image_loader 
import javabridge
import bioformats as bf
javabridge.start_vm(class_path=bf.JARS, max_heap_size='8G', run_headless=True)


fdir = "data/raw/AMP/" 
files = [f for f in os.listdir(fdir) if f[-3:] in ('.vsi')]
images = {}
f = '300-142.vsi'
for f in files:
    print(f)
    image = image_loader.ImageLoaderBioformat(os.path.join(fdir, f),manage_javabridge=False)
    images[f] = image

    # to get a specific region of a scene
    image.get_valid_index() # gets a set of valid scene indices
    image.idx = 11 # set the scene requested
    image.get_region(2000, 2600, 750, 1250, 10) # gets nparray of the region between the boundary (at the original magnification) scaled to `mag`

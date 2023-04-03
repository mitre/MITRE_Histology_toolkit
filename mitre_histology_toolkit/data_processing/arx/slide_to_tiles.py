import matplotlib.pyplot as plt
import numpy as np
import large_image
import skimage.io
import argparse
import slideio
import skimage
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument("filename",
                    help="The name of the whole slide image to tile")
parser.add_argument("config_path",
                    help="The path to the json config file")

args = parser.parse_args()

def process_svs(file_path, filename, output_path, magnification = 20, overlap = 0, tile_size = 1024):
    fpath = os.path.join(file_path, filename)
    main_tag = filename.replace('.svs', '').replace(' ', '_')
    ts = large_image.getTileSource(fpath)
    
    # Create tile folder
    tile_folder = os.path.join(output_path, main_tag)
    if not os.path.exists(tile_folder):
        os.makedirs(tile_folder)
    
    
    tile_iterator = ts.tileIterator(
        scale = dict(magnification = magnification),
        tile_size = dict(width = tile_size, height = tile_size),
        tile_overlap = dict(x = overlap, y = overlap),
        format = large_image.tilesource.TILE_FORMAT_NUMPY)
    
    for tile_info in tile_iterator:
        
        col = tile_info['level_x']
        row = tile_info['level_y']
        
        tile = np.array(tile_info['tile'])[:,:,:3]
        plt.imshow(tile)

        save_name = f'{main_tag}__R{row}_C{col}_TS{tile_size}_OL{overlap}.jpg'
        output_path = os.path.join(tile_folder, save_name)
        skimage.io.imsave(output_path, tile, quality=100)
    return

def process_czi(file_path, filename, output_path, magnification = 20, overlap = 0, tile_size_side = 1024):
    fpath = os.path.join(file_path, filename)
    slide = slideio.open_slide(fpath, "CZI")
    main_tag = filename.replace('.czi', '').replace(' ', '_')
    for scene_id in range(0, slide.num_scenes):
        scene = slide.get_scene(scene_id)
        scene_dim = scene.size
        scene_tag = f'{main_tag}_s{scene_id}'
        
        # Create tile folder
        tile_folder = os.path.join(output_path, scene_tag)
        if not os.path.exists(tile_folder):
            os.makedirs(tile_folder)
        
        for row_index in range(0, scene_dim[0] // tile_size):
            for col_index in range(0, scene_dim[1] // tile_size):
                ys = tile_size * row_index
                xs = tile_size * col_index
                image = scene.read_block((ys, xs, tile_size, tile_size))
                image = image[:,:,::-1]
                save_name = f'{scene_tag}__R{row_index}_C{col_index}_TS{tile_size}_OL{overlap}.jpg'
                save_path = os.path.join(tile_folder, save_name)
                skimage.io.imsave(save_path, image, quality = 100)
    return

if __name__ == '__main__':
    
    # get argparse parameters
    filename = args.filename
    with open(args.config_path, 'r') as json_file:
        configs = json.load(json_file)
    
    # get config parameters
    output_path = f'{configs["root_directory"]}/{configs["tile_directory"]}'
    file_path = f'{configs["root_directory"]}/{configs["slide_directory"]}'
    tile_size = configs["tile_size"]
    overlap = configs["overlap"]
    tissue_thresh = configs["tissue_threshold"]

    file_type = filename[-3:]
    if file_type == 'czi':
        process_svs(file_path, filename, output_path, magnification = 20, overlap = overlap, tile_size = tile_size)
    elif file_type == 'svs':
        process_svs(file_path, filename, output_path, magnification = 20, overlap = overlap, tile_size = tile_size)
    else:
        ValueError(f'Filetype must be czi or svs ({os.path.join(file_path, filename)})')



# skip_scenes=2

# slide = slideio.open_slide(os.path.join(file_path, filename), "SCN")
# main_tag = filename.replace('.scn', '')
# main_tag = main_tag.replace(' ', '_')
# metadata = []

# for ii in range(0, slide.num_scenes):
#     scene = slide.get_scene(ii)
#     meta = {
#         'idx': ii,
#         'rect': scene.rect,
#         'mag':scene.magnification
#         }
#     metadata.append(meta)
    
    
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

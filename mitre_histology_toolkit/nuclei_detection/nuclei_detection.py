#import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from . import slideproc
import pandas as pd
import scipy.sparse
import numpy as np
import skimage
import os

def nuclei_detection(scene_name, tile_path, output_path, scene_metadata, tile_size = 1024, overlap = 0, tissue_thresh = 0.05, foreground_threshold = 140, min_nucleus_area = 6, blood_clot_path = None):
    """
    Performs the entire nuclei detection process, looking for tissue, segmenting the image, and eliminating false detections based on size, shape, and solidity features. The end result is the csv files of position information and the nuclei masks in scipy sparse arrays.

    Parameters
    ----------
    scene_name : str
        The name of the scene being analyzed ({id}_s{scene_id}).
    tile_path : str
        The path to the tiles for a specific whole slide image.
    output_path : str
        The path to the output directory for the csv and scipy sparse arrays.
    scene_metadata : dict
        The metadata dict must have the image resolution and the
        width and height in pixels.
    tile_size : int, optional
        The tile edge size in pixels. The default is 512.
    overlap : int, optional
        The overlap of the tiles in pixels. The default is 0.
    tissue_thresh : float, optional
        The threshold of tissue required for analysis. The default is 0.05.
    foreground_threshold : int, optional
        Threshold for separating foreground and background. The default is 140.
    min_nucleus_area : int, optional
        The minimum size of a nucleus in square microns (mcm^2). The default is 6.
    blood_clot_path : str, optional
        The path to the blood clot npz and json files. If not None, blood clot detections are 
        removed from consideration when detecting nuclei.
    
    Returns
    -------
    Files are saved. No objects are returned.

    """
    filepath = f'{tile_path}/{scene_name}'
    slide = slideproc.process_tiles(filepath,
                                    scene_metadata,
                                    tile_size = tile_size,
                                    overlap = overlap,
                                    tissue_eval = True,
                                    find_nuclei = True,
                                    tissue_thresh = tissue_thresh,
                                    foreground_threshold = foreground_threshold, 
                                    min_nucleus_area = min_nucleus_area,
                                    blood_clot_path = blood_clot_path,
                                    lite = False)
    df = pd.DataFrame(slide)
    df = df[['col', 'row', 'tile_size', 'tissue', 'num_nuclei']]
    output_filepath = f'{output_path}/{scene_name}.csv'
    df.to_csv(output_filepath, index = False)
    
    '''
    Save Nuclei Positions and Segementation Masks
    '''
    nuclei = []
    
    # Create Segmetnation Mask directory
    image_resolution = scene_metadata['resolution']
    fdir = f'{output_path}/{scene_name}'
    px_to_sq_mcm = image_resolution[0] * image_resolution[1]
    px_to_mcm = np.sqrt((image_resolution[0]**2 + image_resolution[1]**2)/2)
    if not os.path.exists(fdir):
        os.makedirs(fdir) 
    for tile in slide:
        if (tile['num_nuclei'] > 0):    
            tile_size = int(tile['tile_size'])
            tile_row = int(tile['row'])
            tile_col = int(tile['col'])
        
            for objProps in tile['nuclei']['objProps']:
                nuclei.append({
                    'tile_size': tile_size,
                    'row': tile_row,
                    'col': tile_col,
                    'nuclei_x_tile': np.round(objProps.centroid[1],5),
                    'nuclei_y_tile': np.round(objProps.centroid[0],5),
                    'nuclei_width': objProps.bbox[3] - objProps.bbox[1] + 1,
                    'nuclei_height': objProps.bbox[2] - objProps.bbox[0] + 1,
                    'nuclei_x_wsi': np.round(objProps.centroid[1] + (tile_col * tile_size),5),
                    'nuclei_y_wsi': np.round(objProps.centroid[0] + (tile_row * tile_size),5),
                    'area': objProps.area * px_to_sq_mcm,
                    'max_intensity': objProps.max_intensity,
                    'min_intensity': objProps.min_intensity,
                    'avg_intensity': objProps.average_intensity,
                    'std_intensity': objProps.std_intensity,
                    'eccentricity': objProps.eccentricity,
                    'major_axis_length': objProps.major_axis_length * px_to_mcm,
                    'minor_axis_length': objProps.minor_axis_length * px_to_mcm,
                    'solidity': objProps.solidity,
                    'label': objProps.label,
                })
    
        row = tile['row']
        col = tile['col']
        tile_size = tile['tile_size']
        overlap = tile['overlap']
        fname = f'{scene_name}__R{row}_C{col}_TS{tile_size}_OL{overlap}_nuclei_mask.npz'
        fpath = f'{fdir}/{fname}'
        try:
            # An empty list is returned when no nuclei is found. 
            im_nuclei_seg_mask = np.copy(tile['nuclei']['im_nuclei_seg_mask'])
        except:
            continue
        #im_nuclei_seg_mask[im_nuclei_seg_mask==1] = 0
        #im_nuclei_seg_mask[im_nuclei_seg_mask>1] = 255
        #im_nuclei_seg_mask = im_nuclei_seg_mask.astype(np.uint8)
        #skimage.io.imsave(fpath, im_nuclei_seg_mask, quality=100)
        #np.save(fpath, im_nuclei_seg_mask)
        scipy.sparse.save_npz(fpath, scipy.sparse.csc_matrix(np.where(im_nuclei_seg_mask == 1, 0, im_nuclei_seg_mask)))
    
    nuclei_df = pd.DataFrame(nuclei)
    output_filepath = f'{fdir}_nuclei_pos.csv'
    nuclei_df.to_csv(output_filepath, index=False)
    
    '''
    Save nuclei segmentation masks
    '''
    # fdir = './output/pickle'
    # if not os.path.exists(fdir):
    #     os.makedirs(fdir) 
    # fname = fdir + '/' + filename + '.p'
    
    # pickle.dump(slide, open(fname,'wb'))
    
    # for tile in slide:
        # fdir = './output/' + filename
        # if not os.path.exists(fdir):
        #     os.makedirs(fdir) 
        # # fname = fdir + '/' + filename + '_R' + str(tile['row']) + '_C' + str(tile['col']) + '_nuclei_mask.tiff'
        # row = tile['row']
        # col = tile['col']
        # tile_size = tile['tile_size']
        # overlap = tile['overlap']
        # fname = 'WS'+ filename + '__R' + str(row) + '_C'+ str(col) + '_TS' + str(tile_size) + '_OL' + str(overlap)+'_nuclei_mask.jpg'
        # fpath = fdir +'/' + fname
        # im_nuclei_seg_mask = np.copy(tile['nuclei']['im_nuclei_seg_mask'])
        # im_nuclei_seg_mask[im_nuclei_seg_mask==1] = 0
        # im_nuclei_seg_mask[im_nuclei_seg_mask>1] = 255
        # im_nuclei_seg_mask = im_nuclei_seg_mask.astype(np.uint8)
        # skimage.io.imsave(fpath, im_nuclei_seg_mask)
    
    #Save Bounding Box Examples
    
    plot_df = df[df['tissue'] > 0.50].sort_values(by = 'tissue', ascending = False)
    
    images = 5
    image_idx = 1
    for tile in slide:
        if (tile['tissue'] > 0.50) and (tile['num_nuclei'] > 50):
            image = tile['tile']
            objProps = tile['nuclei']['objProps']
    
    #         #Bounding Box Visualization
    #         plt.figure(figsize=(5, 5))
    #         plt.imshow(image, origin='lower')
    #         plt.xlim([0, image.shape[1]])
    #         plt.ylim([0, image.shape[0]])
    
    #         for i in range(len(objProps)):
    #             c = [objProps[i].centroid[1], objProps[i].centroid[0], 0]
    #             width = objProps[i].bbox[3] - objProps[i].bbox[1] + 1
    #             height = objProps[i].bbox[2] - objProps[i].bbox[0] + 1
    
    #             cur_bbox = {
    #                 "type":        "rectangle",
    #                 "center":      c,
    #                 "width":       width,
    #                 "height":      height,
    #             }
    
    #             plt.plot(c[0], c[1])
    #             mrect = mpatches.Rectangle([c[0] - 0.5 * width, c[1] - 0.5 * height] ,
    #                                        width, height, fill=False, ec='#00cec9', linewidth=1)
    #             plt.gca().add_patch(mrect)
    #             figname = './output/' + filename + '_R' + str(tile['row']) + '_C' + str(tile['col']) + '_box'
                
    #         #plt.savefig(figname+'.jpg', dpi=300)
    
            #Nuclei Mask Overlay Visualization
            plt.figure(figsize = (5, 5))
            plt.imshow(skimage.color.label2rgb(tile['nuclei']['im_nuclei_seg_mask'], tile['tile'], bg_label=1, alpha=0.5, bg_color=[1,1,1]), origin='lower')
            figname = f'{fdir}_R{tile["row"]}_C{tile["col"]}_overlay.jpg'
            plt.savefig(figname, dpi=300)
    
            #Nuclei Boudnary Visualization
            # im_nuclei_seg_mask = np.copy(tile['nuclei']['im_nuclei_seg_mask'])
            # img = np.copy(tile['tile'])
            # im_nuclei_seg_mask[im_nuclei_seg_mask<2] = 0
            # im_nuclei_seg_mask[im_nuclei_seg_mask>0] = 1
            # im_nuclei_seg_mask = im_nuclei_seg_mask.astype(np.uint8)
            # marker_edge = cv2.Canny(im_nuclei_seg_mask, 1, 1)
            # img[marker_edge == 255] = [0, 206, 201]
            # plt.figure(figsize=(5, 5))
            # plt.imshow(img,origin='lower')
            # figname = './output/' + filename + '_R' + str(tile['row']) + '_C' + str(tile['col']) + '_boundary'
            # plt.savefig(figname+'.jpg', dpi=300)
    
            image_idx += 1
            if(image_idx > images):
                break

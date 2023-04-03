from skimage import morphology, measure
import scipy.sparse
import numpy as np
import slideio
import json
import sys
import cv2
import os

# ============================================================================

CANVAS_SIZE = (600,800)

FINAL_LINE_COLOR = (0, 0, 0)
WORKING_LINE_COLOR = (0, 0, 0)

MINIMUM_TISSUE_AREA = 550000
DISK_RADIUS = 1
MAGNIFICATION = 4
# ============================================================================

class PolygonDrawer(object):
    def __init__(self, window_name, canvas):
        self.window_name = window_name # Name for our window

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.canvas = canvas

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True


    def run(self):
        canvas = self.canvas.copy()
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_NORMAL)#WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, canvas)#np.zeros(CANVAS_SIZE, np.uint8))
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            #canvas = np.zeros(CANVAS_SIZE, np.uint8)
            canvas = self.canvas.copy()
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
                # And  also show what the current segment would look like
                cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        #canvas = np.zeros(CANVAS_SIZE, np.uint8)
        canvas = self.canvas.copy()
        # of a filled polygon
        if (len(self.points) > 0):
            cv2.fillPoly(canvas, np.array([self.points]), FINAL_LINE_COLOR)
        # And show it
        cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        return canvas

def findImageOA_depreceated(slide_num, slide_dir):
    file_exists = False
    for filename in os.listdir(slide_dir):
        if slide_num in filename:
            slide_name = filename
            slide_path = slide_dir + slide_name
            file_exists = True
            break

    print(f'File Loaded: {slide_name}' if file_exists else 'File Not Found')
    return(slide_path, file_exists)

def findImage(slide_num, slide_dir):
    file_exists = False
    for filename in os.listdir(slide_dir):
        file_num = filename.split('_')[0].split('.')[0]
        if slide_num == file_num:
            slide_name = filename
            file_exists = True
            break
        
    print(f'File Loaded: {slide_name}' if file_exists else 'File Not Found')
    if not file_exists:
        slide_name = None
        print('Possible Matches:')
        for filename in os.listdir(slide_dir):
            if slide_num in filename:
                print(slide_num)

    return(slide_name, file_exists)

def loadImage(slide_path, scene_num = 0, magnification = 4):
    ts0 = slideio.open_slide(slide_path, slide_path.split('.')[-1].upper())
    ts = ts0.get_scene(scene_num)
    sx, sy = int(ts.size[0]), int(ts.size[1])
    ratio = magnification / ts.magnification
    new_size = (int(sx * ratio), int(sy * ratio))
    im_low_res = ts.read_block((0, 0, sx, sy), size = new_size)
    
    im_low_res = im_low_res[:,:,::-1] # swap color channels for cv2
    return(im_low_res)

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

def standardize_image_background(scene, intensity_threshold = 200):
    """
    Accounts for scanning artifacts in the scene background. Attempts to set
    all background pixels to pure white. Useful as a pre-processing step 
    before optical density is calculated and tissue is detected. Also sets
    all pitch black areas to white.

    Parameters
    ----------
    scene : numpy array
        The image scene to be updated (often at a low magnification).
    intensity_threshold : int, optional
        The lowest intensity value for any RGB that indicates a pixel is
        background. The default is 200.

    Returns
    -------
    A numpy array at the same resolution as the input array with the background
    pixels set to (255, 255, 255).

    """
    im_copy = scene.copy()
    im_copy[np.where(im_copy.min(axis = 2) > intensity_threshold)] = (255, 255, 255)
    im_copy[np.where(im_copy.max(axis = 2) == 0)] = (255, 255, 255)
    return(im_copy)

def find_tissue(tile, intensity_threshold = 200):
    """
    Segments the tissue and calculates the proportion of the image contaning tissue

    Parameters
    ----------
    tile : numpy array
        RGB image.
    intensity_threshold : int, optional
        The lowest intensity value for any RGB that indicates a pixel is
        background. The default is 200.
    """
    # Account for scanning artifacts in background pixels
    tile = standardize_image_background(tile, intensity_threshold = intensity_threshold)
    
    # Convert to optical density values
    tile = optical_density(tile)
    
    # Threshold at beta and create binary image
    beta = 0.12
    tile = np.max(tile, axis=2) >= beta

    # Remove small holes and islands in the image
    #tile = binary_opening(tile, morphology.disk(3))
    #tile = binary_closing(tile, morphology.disk(3))
    tile = morphology.binary_dilation(tile, morphology.disk(DISK_RADIUS))
    #tile = morphology.binary_fill_holes(tile)
    tile = morphology.binary_erosion(tile, morphology.disk(DISK_RADIUS))
    tile = morphology.remove_small_objects(tile, min_size=MINIMUM_TISSUE_AREA)
    
    # Calculate percentage of tile containig tissue
    percentage = np.mean(tile)
    tissue_amount = percentage #>= tissue_threshold

    return(tissue_amount, tile)

def perform_cuts(tissue_mask, coords):
    im_cut = np.copy(tissue_mask)
    buffer = 0
    
    for ii in range(0, len(coords)-1):
        y1c,x1c = coords[ii]
        y2c,x2c = coords[ii+1]
        
        if buffer > 0:
            for jj in range(-buffer, buffer):
                y1 = y1c + jj
                x1 = x1c + jj
                y2 = y2c + jj
                x2 = x2c + jj
                
                # account for div/0 
                if x2 == x1:
                    x2 += 1
    
                m = (y1-y2)/(x1-x2)
                b = (x1*y2 - x2*y1)/(x1-x2)
    
    
                xx = np.linspace(x1, x2, num=10000)
                yy = m*xx + b
    
    
                for x,y in zip(xx,yy):
                    im_cut[int(x), int(y)] = 0
        else:
            y1 = y1c
            x1 = x1c
            y2 = y2c
            x2 = x2c
    
            # account for div/0 
            if x2 == x1:
                x2 += 1
            
            m = (y1-y2)/(x1-x2)
            b = (x1*y2 - x2*y1)/(x1-x2)
    
    
            xx = np.linspace(x1, x2, num=1000)
            yy = m*xx + b
            
            for x,y in zip(xx,yy):
                im_cut[int(x), int(y)] = 0
    
    return(im_cut)
    
def output_to_files(slide_name, output_dir, im_label, coords, not_synovium_list):
    json_name = slide_name.replace('.svs', '.json')
    sparse_name= slide_name.replace('.svs', '.npz')
    sparse_matrix = scipy.sparse.csc_matrix(im_label)
    scipy.sparse.save_npz(f'{output_dir}/{sparse_name}', sparse_matrix)
    
    annotation = {
        'magnification': MAGNIFICATION,
        'disk_radius': DISK_RADIUS,
        'minimum_area': MINIMUM_TISSUE_AREA,
        'not_synovium': not_synovium_list,
        'cutting_lines': coords
    }
    
    with open(f'{output_dir}/{json_name}', "w") as outfile: 
        json.dump(annotation, outfile)
    

# ============================================================================

if __name__ == "__main__":
    project = sys.argv[1]
    slide_num = sys.argv[2]
    slide_dir = f'data/raw/slides/{project}'
    output_dir = f'data/processed/annotations/{project}/menisci'
    slide_name, file_exists = findImage(slide_num, slide_dir)
    
    if file_exists:
        im_low_res = loadImage(f'{slide_dir}/{slide_name}', scene_num = 0, 
                               magnification = MAGNIFICATION)
        pd = PolygonDrawer("Polygon", im_low_res)
        image = pd.run()
        coords = []
        if len(pd.points) > 0:
            coords = pd.points
        
        print("Polygon = %s" % coords)
        tissue_amount, tissue_mask = find_tissue(im_low_res)
        im_cut = perform_cuts(tissue_mask, coords)
        im_label = measure.label(im_cut, connectivity=1).astype('uint8')
                
        im_label_invert_zero = np.where(im_label == 0, 255, im_label)
        im_label_3 = np.repeat(im_label_invert_zero[:, :, np.newaxis], 3, axis=2).astype('uint8')
        pd = PolygonDrawer("Polygon", im_label_3)
        image = pd.run()
        
        not_synovium_list = []
        not_synovium = pd.points
        if len(not_synovium) > 0:
            for point in not_synovium:
                not_synovium_list += [int(im_label[point[1], point[0]])]
        
        output_to_files(slide_name, output_dir, im_label, coords, sorted(list(set(not_synovium_list))))
        
        
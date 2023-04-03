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

def findImage(slide_num, slide_dir):
    file_exists = False
    for filename in os.listdir(slide_dir):
        if slide_num in filename:
            slide_name = filename
            slide_path = slide_dir + slide_name
            file_exists = True
            break

    print(f'File Loaded: {slide_name}' if file_exists else 'File Not Found')
    return(slide_path, file_exists)

def loadImage(slide_path, magnification = 4):
    ts0 = slideio.open_slide(slide_path, slide_path.split('.')[-1].upper())
    ts = ts0.get_scene(0)
    sx, sy = int(ts.size[0]), int(ts.size[1])
    ratio = magnification / ts.magnification
    new_size = (int(sx * ratio), int(sy * ratio))
    im_low_res = ts.read_block((0, 0, sx, sy), size = new_size)
    
    return(im_low_res[:,:,::-1])

# ============================================================================

if __name__ == "__main__":
    #slide_num = '105_SYN'
    slide_num = sys.argv[1] + '_SYN'
    slide_dir = 'data/raw/slides/HSS_OA/'
    slide_path, file_exists = findImage(slide_num, slide_dir)
    
    if file_exists:
        magnification = 4
        im_low_res = loadImage(slide_path, magnification = magnification)
        pd = PolygonDrawer("Polygon", im_low_res)
        image = pd.run()
        #pd.run()
        #cv2.imwrite("polygon.png", image)#pd.canvas)#
        print("Polygon = %s" % pd.points)
        annotation = {
            'magnification': magnification,
            'blood_clots': pd.points
        }
        
        output_dir = 'data/processed/annotations/blood_clots/'
        output_fpath = slide_path.replace(slide_dir, output_dir).replace('.svs', '_bc_0.json')
        
        rep = 0
        while os.path.isfile(output_fpath):
            output_fpath = output_fpath.replace(f'_bc_{rep}', f'_bc_{rep+1}')
            rep += 1
        
        with open(output_fpath, "w") as outfile: 
            json.dump(annotation, outfile)
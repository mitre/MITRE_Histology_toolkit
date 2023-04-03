import matplotlib.pyplot as plt
from shapely import geometry
import numpy as np
import geojson
import slideio

img_fname = 'data/raw/slides/HSS_RA/45.svs'
filetype = img_fname[-3:].upper()
slide = slideio.open_slide(img_fname, filetype)
scene = slide.get_scene(0)

json_fname = 'data/raw/slides/HSS_RA/45_small.svs.json'

with open(json_fname) as f:
    allobjects = geojson.load(f)

x_coords_all, y_coords_all = [], []
for obj in allobjects:
    xt = [xct[0] for xct in obj['geometry']['coordinates'][0]]
    yt = [yct[1] for yct in obj['geometry']['coordinates'][0]]
    x_coords_all += xt
    y_coords_all += yt

coords = np.c_[x_coords_all, y_coords_all]

xmin, ymin = np.floor(coords.min(axis = 0))
xmax, ymax = np.ceil(coords.max(axis = 0))

xmin = int(xmin)
xmax = int(xmax)
ymin = int(ymin)
ymax = int(ymax)

img = scene.read_block((xmin, ymin, xmax-xmin, ymax-ymin))

updated_objects = []
for obj in allobjects:
    xt = [xct[0] - xmin for xct in obj['geometry']['coordinates'][0]]
    yt = [yct[1] - ymin for yct in obj['geometry']['coordinates'][0]]
    tobj = obj.copy()
    tobj['geometry']['coordinates'] = [list(zip(xt,yt))]
    updated_objects += [tobj]
    
allshapes=[geometry.shape(obj["nucleusGeometry"] if "nucleusGeometry" in obj.keys() else obj["geometry"]) for obj in updated_objects]

x, y = allshapes[0].exterior.coords.xy
print(np.c_[x,y])

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection

# Plots a Polygon to pyplot `ax`
def plot_polygon(ax, poly_list, **kwargs):
    patch_list = []
    for poly in poly_list:
        path = Path.make_compound_path(
            Path(np.asarray(poly.exterior.coords)[:, :2]),
            *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

        patch_list += [PathPatch(path, **kwargs)]
    
    collection = PatchCollection(patch_list, **kwargs)
    
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

fig, ax = plt.subplots()
ax.imshow(img)
plot_polygon(ax, allshapes, facecolor=None, edgecolor='red')
ax.invert_yaxis()
plt.show()


from scipy import sparse
import time
import cv2
import os

def create_geojson_obj(coordinates):
    properties = {"isLocked": False, "object_type": "detection"}
    coord_ring = coordinates.tolist()
    coord_ring += [coord_ring[0]]
    poly = geojson.Polygon([coord_ring])
    return(geojson.feature.Feature(geometry = poly, properties = properties))

image_name = '45_s0'
mask_path = f'data/processed/nuclei/HSS_RA/{image_name}'

st = time.time()
contour_list = []
for mask_fname in os.listdir(mask_path):
    tile_size = 1024
    row = int(mask_fname.split('_R')[1].split('_')[0])
    col = int(mask_fname.split('_C')[1].split('_')[0])
    mask = sparse.load_npz(f'{mask_path}/{mask_fname}').toarray()
    bin_mask = np.where(mask > 0, 1, 0)
    contours, bboxes = cv2.findContours(bin_mask, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if contour.shape[0] > 4:
            ct = np.squeeze(contour)
            ct[:,0] += col*1024
            ct[:,1] += row*1024
            contour_list += [create_geojson_obj(ct)]

print(f'{time.time()-st:.2f}')

st = time.time()

json_annotated_fname = 'data/raw/slides/HSS_RA/nuclei.json'
with open(json_annotated_fname, 'w') as outfile:
    geojson.dump(contour_list, outfile)


print(f'{time.time()-st:.2f}')


#==============================================================================
import cv2
def create_geojson_multi_obj(coordinates, cluster_id):
    # allobjects[int(id)]["properties"]['classification']={'name':classnames[classid],'colorRGB':colors[classid]}
    classes = ['Positive', 'Other', 'Stroma', 'Necrosis', 'Tumor', 'Immune cells', 'Negative']
    colors = {'Positive': -377282,
              'Other': -14336,
              'Stroma': -6895466,
              'Necrosis': -13487566,
              'Tumor': -3670016,
              'Immune cells': -6268256,
              'Negative': -9408287}
    
    properties = {"isLocked": False, "object_type": "detection",
                  "classification": {"name": f'Cluster: {cluster_id}', "colorRGB": colors[classes[cluster_id]]}}
    coord_ring = coordinates.tolist()
    coord_ring += [coord_ring[0]]
    poly = geojson.Polygon([coord_ring])
    return(geojson.feature.Feature(geometry = poly, properties = properties))

image_name = '102_s0'
mag = 4
# slide_loader = 'openslide'
slide_loader = 'slideio'

spx_mask = np.load(f'data/scratch/{image_name}_global_class_mask_{slide_loader}.npz')['arr_0']
img_fname = f'data/raw/slides/HSS_RA/{image_name.replace("_s0", ".svs")}'
filetype = img_fname[-3:].upper()
slide = slideio.open_slide(img_fname, filetype)
scene = slide.get_scene(0)

nx, ny = scene.size
temp_img = scene.read_block((0, 0, nx, ny), size = (nx // (20 // mag), ny // (20 // mag)))
dx = temp_img.shape[1] - spx_mask.shape[1]
dy = temp_img.shape[0] - spx_mask.shape[0]

lx = dx//2  # 300  # 500  # 507
ly = dy-1  # 740  # 300  # 808
plt.imshow(temp_img[ly:(ly-dy), lx:(lx-dx)])
# ((xmin, xmax), (ymin, ymax)) = ((2425, 20324), (4041, 27945))
# xmin = round(xmin / 5)
# xmax = round(xmax / 5)
# ymin = round(ymin / 5)
# ymax = round(ymax / 5)
xmin = lx
xmax = lx + spx_mask.shape[1]
ymin = ly
ymax = ly + spx_mask.shape[0]

new_mask = np.empty((temp_img.shape[0], temp_img.shape[1]))
new_mask[ymin:ymax, xmin:xmax] = cv2.resize(spx_mask, dsize=(xmax - xmin, ymax - ymin),
                                            interpolation=cv2.INTER_NEAREST)
plt.imshow(new_mask);plt.show()
plt.imshow(temp_img)

num_clusters = np.unique(new_mask)[1:]
contour_list = []
contour_sizes = []
for cluster_id in num_clusters:
    bin_mask = np.where(new_mask == cluster_id, 1, 0)
    contours, bboxes = cv2.findContours(bin_mask, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if contour.shape[0] > 4:
            contour_sizes += [max(contour.shape)]
            ct = np.squeeze(contour) * (20 // mag)
            contour_list += [create_geojson_multi_obj(ct, int(cluster_id))]

max_size = sorted(contour_sizes)[-2]
final_list = [contour_list[i0] for i0 in range(len(contour_list)) if contour_sizes[i0] < max_size]

json_annotated_fname = f'data/raw/slides/HSS_RA/spx_{image_name}_{slide_loader}.json'
with open(json_annotated_fname, 'w') as outfile:
    geojson.dump(final_list, outfile)

#==============================================================================
from skimage import transform
import matplotlib.pyplot as plt
from shapely import geometry
import numpy as np
import slideio
import geojson
import cv2
def create_geojson_multi_obj(coordinates, cluster_id):
    # allobjects[int(id)]["properties"]['classification']={'name':classnames[classid],'colorRGB':colors[classid]}
    classes = ['Positive', 'Other', 'Stroma', 'Necrosis', 'Tumor', 'Immune cells', 'Negative']
    colors = {'Positive': -377282,
              'Other': -14336,
              'Stroma': -6895466,
              'Necrosis': -13487566,
              'Tumor': -3670016,
              'Immune cells': -6268256,
              'Negative': -9408287}
    
    properties = {"isLocked": False, "object_type": "detection",
                  "classification": {"name": f'Cluster: {cluster_id}', "colorRGB": colors[classes[cluster_id]]}}
    coord_ring = coordinates.tolist()
    coord_ring += [coord_ring[0]]
    poly = geojson.Polygon([coord_ring])
    return(geojson.feature.Feature(geometry = poly, properties = properties))

image_name = '102_s0'
mag = 4
slide_loader = 'slideio'

spx_mask = np.load(f'data/scratch/{image_name}_exp_masks.npz')['arr_3']
small_spx_mask = np.load(f'data/scratch/{image_name}_global_class_mask_{slide_loader}.npz')['arr_0']
spx_mask = cv2.resize(spx_mask, dsize=(small_spx_mask.shape[1], small_spx_mask.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
img_fname = f'data/raw/slides/HSS_RA/{image_name.replace("_s0", ".svs")}'
filetype = img_fname[-3:].upper()
slide = slideio.open_slide(img_fname, filetype)
scene = slide.get_scene(0)

nx, ny = scene.size
temp_img = scene.read_block((0, 0, nx, ny), size = (nx // (20 // mag), ny // (20 // mag)))
dx = temp_img.shape[1] - spx_mask.shape[1]
dy = temp_img.shape[0] - spx_mask.shape[0]

lx = dx//2 - 10 # 300  # 500  # 507
ly = dy-60 # 740  # 300  # 808
# plt.imshow(temp_img[ly:(ly-dy), lx:(lx-dx)])

xmin = lx
xmax = lx + spx_mask.shape[1]
ymin = ly
ymax = ly + spx_mask.shape[0]

new_mask = np.empty((temp_img.shape[0], temp_img.shape[1]))
new_mask[ymin:ymax, xmin:xmax] = cv2.resize(spx_mask, dsize=(xmax - xmin, ymax - ymin),
                                            interpolation=cv2.INTER_NEAREST)

# plt.imshow(new_mask);plt.show()
# plt.imshow(temp_img)

num_clusters = np.unique(new_mask)[1:]
contour_list = []
contour_sizes = []
for cluster_id in num_clusters:
    bin_mask = np.where(new_mask == cluster_id, 1, 0)
    contours, bboxes = cv2.findContours(bin_mask, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if contour.shape[0] > 4:
            contour_sizes += [max(contour.shape)]
            ct = np.squeeze(contour) * (20 // mag)
            contour_list += [create_geojson_multi_obj(ct, int(cluster_id))]

max_size = sorted(contour_sizes)[-2]
final_list = [contour_list[i0] for i0 in range(len(contour_list)) if contour_sizes[i0] < max_size]

json_annotated_fname = f'data/raw/slides/HSS_RA/spx_{image_name}_individual.json'
with open(json_annotated_fname, 'w') as outfile:
    geojson.dump(final_list, outfile)
# ========================================
# run at mag 4 instead 
color_list = ['red', 'blue', 'magenta', 'cyan', 'black']
color_dict = {cluster_id: color_list[int(cluster_id - 1)] for cluster_id in num_clusters}
shape_dict = {}
for cluster_id in num_clusters:
    shape_dict[cluster_id] = []
    bin_mask = np.where(new_mask == cluster_id, 1, 0)
    contours, bboxes = cv2.findContours(bin_mask, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if contour.shape[0] > 4:
            ct = np.squeeze(contour)# * (20 // mag)
            shape_dict[cluster_id] += [geometry.shape(create_geojson_multi_obj(ct, int(cluster_id))['geometry'])]

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection

# Plots a Polygon to pyplot `ax`
def plot_polygon(ax, poly_list, **kwargs):
    patch_list = []
    for poly in poly_list:
        path = Path.make_compound_path(
            Path(np.asarray(poly.exterior.coords)[:, :2]),
            *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

        patch_list += [PathPatch(path, **kwargs)]
    
    collection = PatchCollection(patch_list, **kwargs)
    
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

fig, ax = plt.subplots()
ax.imshow(temp_img)
for cluster_id in num_clusters:
    plot_polygon(ax, shape_dict[cluster_id], facecolor=color_dict[cluster_id], edgecolor=color_dict[cluster_id])
plt.show()

##########################################################


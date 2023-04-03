import os
os.chdir('mitre_histology_toolkit/data_processing')
import DLTrace
os.chdir('../..')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

nuc_dir = 'data/processed/nuclei/HSS_RA'
clst_dir = 'data/processed/clusters/HSS_RA/convex'
img_name = '247_s0'

nuc_df = pd.read_csv(f'{nuc_dir}/{img_name}_nuclei_pos.csv')
clst_df = pd.read_csv(f'{clst_dir}/{img_name}_dtfe_cluster_key.csv')
x = nuc_df['nuclei_x_wsi']
y = nuc_df['nuclei_y_wsi']

myDL = DLTrace.DLTrace(x, y, buffer=1.0, DTFE=True, num_stdev=1)
myDL.run()


fig, ax = plt.subplots(ncols = 2, figsize = (8,4), sharex = True, sharey = True)
ax[0].scatter(x, y, s = .3, c = 'k')
ax[0].scatter(clst_df['nuclei_x_wsi'], clst_df['nuclei_y_wsi'], s = .5, c = '#1f76b4')

area_min = np.exp(12)  # 4e5
ax[1].scatter(x, y, s = .3, c = 'k')
myDL.traces[myDL.traces.area > area_min].plot(ax = ax[1])

ax[0].invert_yaxis()
for axi in ax:
    axi.set_aspect(1)

ax[0].set_title('DTFE Aggregate Detection')
ax[1].set_title('Distance Based DLTrace')


#########################################################################

import os
os.chdir('mitre_histology_toolkit/data_processing')
import DLTrace
os.chdir('../..')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import slideio

nuc_dir = 'data/processed/nuclei/HSS_RA'
dtfe_dir = 'data/processed/node_features/dtfe/base/HSS_RA'
img_dir = 'data/raw/slides/HSS_RA'
img_name = '247_s0'
mag = 4

nuc_df = pd.read_csv(f'{nuc_dir}/{img_name}_nuclei_pos.csv')
dtfe_df = pd.read_csv(f'{dtfe_dir}/{img_name}_base_dtfe.csv')
slide = slideio.Slide(f'{img_dir}/{img_name.replace("_s0", ".svs")}', "SVS")
scene = slide.get_scene(0)
full_width, full_height = scene.size
ds_rate = (scene.magnification / mag)
ds_width = int(full_width / ds_rate)
ds_height = int(full_height / ds_rate)

img = scene.read_block((0, 0, full_width, full_height), size = (ds_width, ds_height))

nuc_df['axis_ratio'] = nuc_df.major_axis_length / nuc_df.minor_axis_length

nuc_sub_df = nuc_df[(nuc_df.major_axis_length < 13) &
                    #(nuc_df.area < 200) &
                    (nuc_df.axis_ratio < 1.4) &
                    (nuc_df.eccentricity < .6)]

fig, ax = plt.subplots(ncols = 2, figsize = (8,4), sharex = True, sharey = True)
ax[0].scatter(dtfe_df['nuclei_x_wsi'], dtfe_df['nuclei_y_wsi'], s = .5, 
              c = dtfe_df['dtfe'], cmap = 'jet', vmax = 0.007)

ax[1].scatter(nuc_df['nuclei_x_wsi'], nuc_df['nuclei_y_wsi'], s = .3, c = 'k')
ax[1].scatter(nuc_sub_df['nuclei_x_wsi'], nuc_sub_df['nuclei_y_wsi'], s = .3, c = 'b')

ax[0].invert_yaxis()
for axi in ax:
    axi.set_aspect(1)

ax[0].set_title('DTFE Aggregate Detection')
ax[1].set_title(r'Nuclei Remaining: $\frac{%s}{%s} = %.3f$' % (nuc_sub_df.shape[0], nuc_df.shape[0], nuc_sub_df.shape[0] / nuc_df.shape[0]))

fig, ax = plt.subplots(figsize = (8,8))
ax.imshow(img)
ax.scatter(nuc_sub_df['nuclei_x_wsi'] / ds_rate, nuc_sub_df['nuclei_y_wsi'] / ds_rate, s = .003, c = 'lightblue')
fig.savefig('test.png', dpi = 1500)


#######
myDL = DLTrace.DLTrace(x, y, buffer=1.0, DTFE=True, num_stdev=1)
myDL.run()

fig, ax = plt.subplots(ncols = 2, figsize = (8,4), sharex = True, sharey = True)
ax[0].scatter(dtfe_df['nuclei_x_wsi'], dtfe_df['nuclei_y_wsi'], s = .5, 
              c = dtfe_df['dtfe'], cmap = 'jet', vmax = 0.007)

area_min = np.exp(12)  # 4e5
ax[1].scatter(x, y, s = .3, c = 'k')
myDL.traces[myDL.traces.area > area_min].plot(ax = ax[1])

ax[0].invert_yaxis()
for axi in ax:
    axi.set_aspect(1)

ax[0].set_title('DTFE Aggregate Detection')
ax[1].set_title('Distance Based DLTrace')

#####################################################################

import matplotlib.pyplot as plt
import scipy.sparse
import pandas as pd
import numpy as np
import skimage.io
import slideio
import skimage

img_name = '45_s0'  # '247_s0'  # '87_s0'
nuc_dir = 'data/processed/nuclei/HSS_RA'
dtfe_dir = 'data/processed/node_features/dtfe/base/HSS_RA'
mask_dir = f'{nuc_dir}/{img_name}'
tile_dir = mask_dir.replace('nuclei', 'tiles')
img_dir = 'data/raw/slides/HSS_RA'
mag = 4

dtfe_df = pd.read_csv(f'{dtfe_dir}/{img_name}_base_dtfe.csv')
slide = slideio.Slide(f'{img_dir}/{img_name.replace("_s0", ".svs")}', "SVS")
scene = slide.get_scene(0)
full_width, full_height = scene.size
ds_rate = (scene.magnification / mag)
ds_width = int(full_width / ds_rate)
ds_height = int(full_height / ds_rate)

# img = scene.read_block((0, 0, full_width, full_height), size = (ds_width, ds_height))


row = 5  # 18  # 10
col = 12  # 19  # 3

nuc_df = pd.read_csv(f'{nuc_dir}/{img_name}_nuclei_pos.csv')
nuc_df = nuc_df[(nuc_df.row == row) &
                (nuc_df.col == col)]

nuc_df['axis_ratio'] = nuc_df.major_axis_length / nuc_df.minor_axis_length

nuc_area = 250
axis_ratio = 1.8
eccentricity = 0.9
solidity = 0.8
nuc_sub_df = nuc_df[(nuc_df.area < nuc_area) &
                    (nuc_df.axis_ratio < axis_ratio) &
                    (nuc_df.eccentricity < eccentricity) &
                    (nuc_df.solidity > solidity)]

nuc_rm = nuc_df.label[np.isin(nuc_df.label, nuc_sub_df.label, invert = True)]

mask = scipy.sparse.load_npz(f'{mask_dir}/{img_name}__R{row}_C{col}_TS1024_OL0_nuclei_mask.npz').toarray()
tile = skimage.io.imread(f'{tile_dir}/{img_name}__R{row}_C{col}_TS1024_OL0.jpg')
gs_tile = skimage.color.rgb2gray(tile)

mask_sub = np.where(np.isin(mask, nuc_sub_df.label), mask, 0)
mask_rm = np.where(np.isin(mask, nuc_rm), mask, 0)

# fig, ax = plt.subplots(ncols = 3, figsize = (15, 5))
# ax[0].imshow(skimage.color.label2rgb(mask_rm, gs_tile, bg_label=0, alpha=0.5, bg_color=[1,1,1]))
# ax[1].imshow(skimage.color.label2rgb(mask, gs_tile, bg_label=0, alpha=0.5, bg_color=[1,1,1]))
# ax[2].imshow(skimage.color.label2rgb(mask_sub, gs_tile, bg_label=0, alpha=0.5, bg_color=[1,1,1]))

# for axi in ax:
#     axi.set_xticks([])
#     axi.set_yticks([])

# plt.tight_layout()


fig, ax = plt.subplots(ncols = 3, figsize = (9, 3.5))
ax[0].imshow(skimage.color.label2rgb(mask_rm, gs_tile, bg_label=0, alpha=0.5, bg_color=[1,1,1]))
ax[1].imshow(skimage.color.label2rgb(mask, gs_tile, bg_label=0, alpha=0.5, bg_color=[1,1,1]))
ax[2].imshow(skimage.color.label2rgb(mask_sub, gs_tile, bg_label=0, alpha=0.5, bg_color=[1,1,1]))

for axi in ax:
    axi.set_xticks([])
    axi.set_yticks([])
    axi.set_xlim([0, 128])
    axi.set_ylim([0, 128])

fig.suptitle(f'Area < {nuc_area}, Axis Ratio < {axis_ratio}, Eccentricity < {eccentricity}, Solidity > {solidity}')
ax[0].set_title('Omitted')
ax[1].set_title('All')
ax[2].set_title('Included')
plt.tight_layout()

rp = skimage.measure.regionprops(mask[0:128,0:128])
sol_list = []
nm_list = []
for rrp in rp:
    image = np.zeros((rrp.image.shape[0] + 10, rrp.image.shape[1] + 10)).astype(int)
    image[5:-5,5:-5] = rrp.image
    region = skimage.measure.regionprops(image)[0]
    
    r, c = region.centroid
    r_radius = region.axis_minor_length / 2
    c_radius = region.axis_major_length / 2
    rotation = np.pi/2 + region.orientation
    
    rr, cc = skimage.draw.ellipse(r, c, r_radius, c_radius, shape=image.shape, rotation=rotation)
    
    img2 = np.zeros(image.shape).astype(int)
    img2[rr, cc] = 2

    plt.imshow(img2 + image)
    new_metric = (img2 * image).sum() / img2.sum()
    plt.title(f'Solidity: {rrp.solidity:.2f}, Eccentricity: {rrp.eccentricity:.2f}, New Metric: {new_metric:.2f}')
    plt.show()
    sol_list += [rrp.solidity]
    nm_list += [new_metric]

plt.scatter(sol_list, nm_list)
plt.plot([min(sol_list + nm_list), 1], [min(sol_list + nm_list), 1])


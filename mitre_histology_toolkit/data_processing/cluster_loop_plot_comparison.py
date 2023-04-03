import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skimage.io
pd.set_option('display.max_columns', None)

image_name = '120_s0'
data_dir = 'data/processed/clusters/HSS_RA/multiple_params/convex'

clst_key_df = pd.read_csv(f'{data_dir}/{image_name}_dtfe_cluster_key.csv')
clst_df = pd.read_csv(f'{data_dir}/{image_name}_dtfe_clusters.csv')
wsi_df = pd.read_csv(f'{data_dir}/{image_name}_dtfe_wsi.csv')

nuc_df = pd.read_csv(f'data/processed/nuclei/HSS_RA/{image_name}_nuclei_pos.csv')

print(wsi_df[['dtfe_threshold', 'number_nodes_necessary_for_cluster', 'num_clusters', 'numPixels']])
###############################################################################
fig, ax = plt.subplots(figsize = (8,8))
ax.scatter(nuc_df.nuclei_x_wsi, nuc_df.nuclei_y_wsi, s = 0.1)
for dtfe_threshold in [0.003]:#[0.006999999999999999]:#[0.009000000000000001]:
    for nnnfc in [10, 30, 50, 100, 200]:
        sub_df = clst_key_df[(clst_key_df.threshold == dtfe_threshold) & (clst_key_df.number_nodes_necessary_for_cluster == nnnfc)].reset_index(drop = True)
        if sub_df.shape[0] > 0:
            ax.scatter(sub_df.nuclei_x_wsi, sub_df.nuclei_y_wsi, s = 1, label = f'{nnnfc} Nodes')

ax.set_aspect(1)
ax.invert_yaxis()
legend = ax.legend()
#change the marker size manually for both lines
for lgh in legend.legendHandles:
    lgh._sizes = [30]
###############################################################################
fig, ax = plt.subplots(figsize = (8,8))
ax.scatter(nuc_df.nuclei_x_wsi, nuc_df.nuclei_y_wsi, s = 0.1)
for nnnfc in [100]:
    for dtfe_threshold in np.unique(clst_key_df.threshold):
        sub_df = clst_key_df[(clst_key_df.threshold == dtfe_threshold) & (clst_key_df.number_nodes_necessary_for_cluster == nnnfc)].reset_index(drop = True)
        if sub_df.shape[0] > 0:
            ax.scatter(sub_df.nuclei_x_wsi, sub_df.nuclei_y_wsi, s = 1, label = f'DTFE = {dtfe_threshold:.3f}')

ax.set_aspect(1)
ax.invert_yaxis()
ax.set_title(f'At Least {nnnfc} Nodes per Aggregate')
legend = ax.legend()
#change the marker size manually for both lines
for lgh in legend.legendHandles:
    lgh._sizes = [30]
###############################################################################

sub_df.merge(nuc_df)


fig, ax = plt.subplots(figsize = (8,8))
r1, c1 = 6, 10
r2, c2 = 10, 10
ax.scatter(nuc_df.nuclei_x_wsi, nuc_df.nuclei_y_wsi, s = 1)
ax.scatter(nuc_df.nuclei_x_wsi[(nuc_df.row == r1) & (nuc_df.col == c1)], nuc_df.nuclei_y_wsi[(nuc_df.row == r1) & (nuc_df.col == c1)], s = 1)
ax.scatter(nuc_df.nuclei_x_wsi[(nuc_df.row == r2) & (nuc_df.col == c2)], nuc_df.nuclei_y_wsi[(nuc_df.row == r2) & (nuc_df.col == c2)], s = 1)
ax.set_aspect(1)
ax.invert_yaxis()

rr, cc = r1, c1
tile = skimage.io.imread(f'data/processed/tiles/HSS_RA/{image_name}/{image_name}__R{rr}_C{cc}_TS1024_OL0.jpg')
xpos = nuc_df.nuclei_x_wsi[(nuc_df.row == rr) & (nuc_df.col == cc)] - 1024 * cc
ypos = nuc_df.nuclei_y_wsi[(nuc_df.row == rr) & (nuc_df.col == cc)] - 1024 * rr
fig, ax = plt.subplots(figsize = (8,8))
ax.imshow(tile)
ax.scatter(xpos, ypos, s = 3, c = 'yellow')
ax.set_aspect(1)




dtfe_df = pd.read_csv(f'data/processed/node_features/dtfe/base/HSS_RA/{image_name}_base_dtfe.csv')
fig, ax = plt.subplots(figsize = (6,6))
ax.scatter(dtfe_df.nuclei_x_wsi, dtfe_df.nuclei_y_wsi, s = 0.1, c = dtfe_df.dtfe, cmap = 'jet', vmin = 0, vmax = 0.002)
ax.set_aspect(1)
ax.invert_yaxis()
ax.set_title('Heatmap of DTFE')

sub_df = clst_key_df[(clst_key_df.threshold == np.unique(clst_key_df.threshold)[2]) & (clst_key_df.number_nodes_necessary_for_cluster == 100)].reset_index(drop = True)

fig, ax = plt.subplots(figsize = (6,6))
ax.scatter(nuc_df.nuclei_x_wsi, nuc_df.nuclei_y_wsi, s = 0.1)
ax.scatter(sub_df.nuclei_x_wsi, sub_df.nuclei_y_wsi, s = 1, c = 'darkred')
ax.set_aspect(1)
ax.invert_yaxis()
ax.set_title('Clusters Retained')


import matplotlib.pyplot as plt
import pandas as pd

dtfe_dir = 'data/processed/node_features/dtfe/AMP'
clst_dir = 'data/processed/clusters/AMP'
nuc_dir = 'data/processed/output/AMP'
#img = '300-0175_Right_Wrist'
img = '300-489_s1'
dtfe_df = pd.read_csv(f'{dtfe_dir}/{img}_base_dtfe.csv')
fig, ax = plt.subplots()
ax.scatter(dtfe_df.nuclei_x_wsi, dtfe_df.nuclei_y_wsi, s = 1, c = dtfe_df.dtfe, 
           cmap = 'jet', vmax = 0.002)
ax.set_aspect(1)
# xmin, xmax = (9000, 11500)
# ymin, ymax = (12500, 15000)
# ax.set_xlim((xmin, xmax))
# ax.set_ylim((ymin, ymax))
ax.invert_yaxis()


clst_df = pd.read_csv(f'{clst_dir}/{img}_dtfe_cluster_key.csv')
fig, ax = plt.subplots(figsize = (10,10))
ax.scatter(dtfe_df.nuclei_x_wsi, dtfe_df.nuclei_y_wsi, s = 1)
ax.scatter(clst_df.nuclei_x_wsi, clst_df.nuclei_y_wsi, s = 1)
ax.set_aspect(1)
# xmin, xmax = (9000, 11500)
# ymin, ymax = (12500, 15000)
# ax.set_xlim((xmin, xmax))
# ax.set_ylim((ymin, ymax))
ax.invert_yaxis()

from sklearn import decomposition
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import os
pd.set_option('display.max_columns', None)

@st.cache
def load_data(xcols):
    proj_list = ['HSS_RA', 'HSS_OA']
    df_list_full = []
    for proj in proj_list:
        clst_dir_name = f'data/processed/clusters/{proj}/multiple_params/convex'
        nuc_dir_name = f'data/processed/nuclei/{proj}'
        df_list = []
        for fname in os.listdir(clst_dir_name):
            if 'wsi' in fname:
                fname_base = fname.replace('_dtfe_wsi.csv', '')
                xwsi = pd.read_csv(f'{clst_dir_name}/{fname}')
                xcls = pd.read_csv(f'{clst_dir_name}/{fname_base}_dtfe_clusters.csv')
                xtis = pd.read_csv(f'{nuc_dir_name}/{fname_base}.csv')
                
                xtis = xtis[xtis['tissue'] > 0.1].reset_index(drop = True)
                xwsi['avg_tissue_per_tile'] = xtis.tissue.mean()
                xwsi['num_tiles'] = xtis.shape[0]
                xwsi['original_num_nuclei'] = xtis.num_nuclei.sum()
                
                xwsi = xwsi.merge(xcls, how = 'left', suffixes = ['_total', '_cluster'],
                                  left_on = ['ID', 'dtfe_threshold', 'edge_range', 'number_nodes_necessary_for_cluster'],
                                  right_on = ['ID', 'threshold', 'edge_range', 'number_nodes_necessary_for_cluster'])
                df_list += [xwsi]
                
        tdf = pd.concat(df_list)
        tdf['project']= proj
        
        df_list_full += [tdf]
    
    wsi_df = pd.concat(df_list_full)
    wsi_df['project'] = wsi_df['project'].astype('category')
    wsi_df = wsi_df[['ID', 'dtfe_threshold', 'edge_range',
                     'number_nodes_necessary_for_cluster', 'bc_frac_dim_dtfe_wsi',
                     'bc_frac_dim_full_wsi', 'num_clusters', 'numPixels_total',
                     'numNuclei_total', 'numNucleiDtfeOnly', 'avg_tissue_per_tile',
                     'num_tiles', 'original_num_nuclei', 'cluster_ID',
                     'convex_area', 'concave_area', 'eccentricity', 'numPixels_cluster',
                     'numNuclei_cluster', 'major_axis_length', 'minor_axis_length',
                     'bc_frac_dim_cluster', 'project']]
    
    wsi_df.columns = ['ID', 'dtfe_threshold', 'edge_range',
                     'number_nodes_necessary_for_cluster', 'bc_frac_dim_dtfe_wsi',
                     'bc_frac_dim_full_wsi', 'num_clusters', 'num_pixels_total',
                     'num_nuclei_total', 'num_nuclei_dtfe_only', 'avg_tissue_per_tile',
                     'num_tiles', 'original_num_nuclei', 'cluster_ID',
                     'convex_area', 'concave_area', 'eccentricity', 'num_pixels_cluster',
                     'num_nuclei_cluster', 'major_axis_length', 'minor_axis_length',
                     'bc_frac_dim_cluster', 'project']
    
    xdf = wsi_df[['project', 'ID', 'dtfe_threshold', 'edge_range',
                  'number_nodes_necessary_for_cluster'] + xcols].groupby(['project', 'ID', 'dtfe_threshold', 'edge_range',
                  'number_nodes_necessary_for_cluster'])[xcols].max().reset_index()
    return(wsi_df, xdf)

@st.cache
def get_xdict(xdf):
    xdict = {}
    for dtfe in np.round(np.unique(xdf.dtfe_threshold), 3):
        if dtfe not in xdict:
            xdict[dtfe] = {}
        for edge_range in np.unique(xdf.edge_range):
            if edge_range not in xdict[dtfe]:
                xdict[dtfe][edge_range] = {}
            for nnodes in np.unique(xdf.number_nodes_necessary_for_cluster):
                if nnodes not in xdict[dtfe][edge_range]:
                    xdict[dtfe][edge_range][nnodes] = {'X': {}, 'ID': {}}
                    
                
                sub_df_alt = xdf[(xdf.dtfe_threshold.round(3) == dtfe) &
                             (xdf.edge_range == edge_range) &
                             (xdf.number_nodes_necessary_for_cluster == nnodes)]
                
                xdict[dtfe][edge_range][nnodes]['X'] = np.array(sub_df_alt[xcols])
                xdict[dtfe][edge_range][nnodes]['ID'] = sub_df_alt[['ID']].reset_index(drop = True)
    
    return(xdict)

def scatterplot(sub_df, xstr, ystr):
    return(sns.scatterplot(data = sub_df, x = xstr, y = ystr, hue = 'project', alpha = 0.5).figure)
    #st.pyplot(g.figure)
    #return(g.figure)

def boxplot(sub_df, ystr, violin = False):
    if violin:
        return(sns.violinplot(data = sub_df, x = 'project', y = ystr, hue = 'project').figure)
    else:
        return(sns.boxplot(data = sub_df, x = 'project', y = ystr, hue = 'project').figure)


def pcaplot(pca, X, y_labels, x_component_index, y_component_index):
    ypca_x = X.dot(pca.components_[x_component_index])
    ypca_y = X.dot(pca.components_[y_component_index])
    temp_df = pd.DataFrame({'project': y_labels, 'pca_x': ypca_x, 'pca_y': ypca_y})
    
    g = sns.scatterplot(data = temp_df, x = 'pca_x', y = 'pca_y', hue = 'project')
    g.axes.set_title('Principal Component Analysis')
    g.axes.set_xlabel(f'PCA {x_component_index + 1}: {pca.explained_variance_ratio_[x_component_index] * 100:.1f}%')
    g.axes.set_ylabel(f'PCA {y_component_index + 1}: {pca.explained_variance_ratio_[y_component_index] * 100:.1f}%')
    return(g.figure)

# xcols = ['bc_frac_dim_dtfe_wsi',
#             'bc_frac_dim_full_wsi', 'num_clusters', 'num_pixels_total',
#             'num_nuclei_total', 'num_nuclei_dtfe_only', 'avg_tissue_per_tile',
#             'num_tiles', 'original_num_nuclei',
#             'convex_area', 'concave_area', 'eccentricity', 'num_pixels_cluster',
#             'num_nuclei_cluster', 'major_axis_length', 'minor_axis_length',
#             'bc_frac_dim_cluster']

#xcols = ['num_clusters', 'num_nuclei_cluster']

xcols = ['bc_frac_dim_dtfe_wsi',
            'bc_frac_dim_full_wsi', 'num_clusters', 'num_pixels_total',
            'num_nuclei_total', 'num_nuclei_dtfe_only', 'avg_tissue_per_tile',
            'num_tiles', 'original_num_nuclei',
            'convex_area', 'concave_area', 'eccentricity', 'num_pixels_cluster',
            'num_nuclei_cluster', 'major_axis_length', 'minor_axis_length',
            'bc_frac_dim_cluster']

wsi_df, xdf = load_data(xcols)
xdict = get_xdict(xdf)

dtfe_thresholds = st.sidebar.radio('DTFE Threshold', np.round(np.unique(xdf.dtfe_threshold), 3),
                           on_change = None)
num_nodes = st.sidebar.radio('Min Number of Nodes', np.unique(xdf.number_nodes_necessary_for_cluster),
                     on_change = None)

X = xdict[dtfe_thresholds][100][num_nodes]['X'].copy()
X[np.where(np.isnan(X))] = 0
idx = xdict[dtfe_thresholds][100][num_nodes]['ID']
y = idx.merge(wsi_df[['ID', 'project']].drop_duplicates())['project']
pca = decomposition.PCA(n_components=5)
pca.fit(X)
st.pyplot(pcaplot(pca, X, y, 0, 1), clear_figure = True)

sub_df = xdf[(xdf.dtfe_threshold.round(3) == dtfe_thresholds) &
             (xdf.number_nodes_necessary_for_cluster == num_nodes)]

sub_df['log_num_clusters'] = np.log(sub_df['num_clusters'])
sub_df['log_num_nuclei_cluster'] = np.log(sub_df['num_nuclei_cluster'])
sub_df['log_convex_area'] = np.log(sub_df['convex_area'])
sub_df['log_concave_area'] = np.log(sub_df['concave_area'])
sub_df['log_num_pixels_cluster'] = np.log(sub_df['num_pixels_cluster'])
sub_df['total_tissue'] = sub_df['avg_tissue_per_tile'] * sub_df['num_tiles']
sub_df['norm_max_cluster'] = sub_df['num_pixels_cluster'] / sub_df['total_tissue']
sub_df['norm_num_clusters'] = sub_df['num_clusters'] / sub_df['total_tissue']
#st.table(sub_df.head())

st.pyplot(scatterplot(sub_df, 'num_clusters', 'num_nuclei_cluster'), clear_figure = True)
st.pyplot(scatterplot(sub_df, 'log_num_clusters', 'log_num_nuclei_cluster'), clear_figure = True)
st.pyplot(scatterplot(sub_df, 'log_convex_area', 'log_concave_area'), clear_figure = True)
st.pyplot(scatterplot(sub_df, 'log_num_clusters', 'log_num_nuclei_cluster'), clear_figure = True)
st.pyplot(scatterplot(sub_df, 'num_nuclei_cluster', 'num_pixels_cluster'), clear_figure = True)
st.pyplot(scatterplot(sub_df, 'log_num_nuclei_cluster', 'log_num_pixels_cluster'), clear_figure = True)
st.pyplot(scatterplot(sub_df, 'norm_num_clusters', 'norm_max_cluster'), clear_figure = True)


st.pyplot(boxplot(sub_df, 'num_clusters', violin = True), clear_figure = True)
st.pyplot(boxplot(sub_df, 'log_num_nuclei_cluster', violin = True), clear_figure = True)



















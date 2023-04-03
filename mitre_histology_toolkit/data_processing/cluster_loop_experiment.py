import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import time
pd.set_option('display.max_columns', None)

def get_wsi_df(proj_list, inclusion_list):
    df_list_full = []
    for proj in proj_list:
        clst_dir_name = f'data/processed/clusters/{proj}/multiple_params/convex'
        nuc_dir_name = f'data/processed/nuclei/{proj}'
        df_list = []
        for fname in os.listdir(clst_dir_name):
            if 'wsi' in fname:
                #if (proj == 'HSS_OA') or (fname.split('_dtfe')[0] in ra_knee_ids):
                if fname.split('_dtfe')[0] in inclusion_list:
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
    wsi_df.fillna(0, inplace = True)
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
    
    wsi_df['total_tissue'] = wsi_df['avg_tissue_per_tile'] * wsi_df['num_tiles']
    wsi_df['nuclei_density'] = wsi_df['original_num_nuclei'] / wsi_df['total_tissue']
    wsi_df['nuclei_density_aggregates'] = wsi_df['num_nuclei_total'] / wsi_df['total_tissue']
    wsi_df['nuclei_density_dtfe'] = wsi_df['num_nuclei_dtfe_only'] / wsi_df['total_tissue']
    wsi_df['norm_pixels_max_cluster'] = wsi_df['num_pixels_cluster'] / wsi_df['total_tissue']
    wsi_df['norm_nuclei_max_cluster'] = wsi_df['num_nuclei_cluster'] / wsi_df['total_tissue']
    wsi_df['norm_num_clusters'] = wsi_df['num_clusters'] / wsi_df['total_tissue']
    wsi_df['bc_frac_dim_wsi_ratio'] = wsi_df['bc_frac_dim_dtfe_wsi'] / wsi_df['bc_frac_dim_full_wsi']
    wsi_df['norm_bc_frac_dim_dtfe_wsi'] = wsi_df['bc_frac_dim_dtfe_wsi'] / wsi_df['total_tissue']
    
    return(wsi_df)

def get_classification_data(wsi_df, yvar, classes, xcols, dtfe, edge_range, nnodes):
    xdf = wsi_df[[yvar, 'ID', 'dtfe_threshold', 'edge_range',
                  'number_nodes_necessary_for_cluster'] + xcols].groupby([yvar, 'ID', 'dtfe_threshold', 'edge_range',
                  'number_nodes_necessary_for_cluster'])[xcols].max().reset_index()
    
    sub_df = xdf[(xdf.dtfe_threshold.round(4) == dtfe) &
                 (xdf.edge_range == edge_range) &
                 (xdf.number_nodes_necessary_for_cluster == nnodes)]
                
    X = np.array(sub_df[xcols])
    X[np.where(np.isnan(X))] = 0
    y = label_binarize(np.array(sub_df[yvar]), classes = classes).flatten()
    return(X, y)

def get_oa_ra_classification_data(wsi_df, xcols, dtfe, edge_range, nnodes):
    yvar = 'project'
    classes = ['HSS_RA', 'HSS_OA']
    return(get_classification_data(wsi_df, yvar, classes, xcols, dtfe, edge_range, nnodes))

def perform_test(X, y, nreps, xcols, model = 'rf'):
    auc_list = []
    acc_list = []
    f1_list = []
    fm_list = []
    for i in range(nreps):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y)

        if model == 'rf':
            model_object = RandomForestClassifier(n_estimators = 10, max_depth = None, min_samples_split = 2)
        elif model == 'lda':
            model_object = LinearDiscriminantAnalysis()
        elif model == 'logit':
            model_object = LogisticRegression()
        
        model_object.fit(X_train, y_train)
        
        result = permutation_importance(
            model_object, X_test, y_test, n_repeats=10, n_jobs=2
        )
        
        if i == 0:
            forest_importances = result['importances'].T
        else:
            forest_importances = np.r_[forest_importances, result['importances'].T]
        
        y_score = model_object.predict_proba(X_test)[:,1]
        y_score_labels = model_object.predict(X_test)
        
        f1score = metrics.f1_score(y_test, y_score_labels)
        fmscore = metrics.fowlkes_mallows_score(y_test, y_score_labels)
        
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        auc_list += [roc_auc]
        acc_list += [(y_score_labels == y_test).sum() / len(y_test)]
        f1_list += [f1score]
        fm_list += [fmscore]
    
    forest_importances_df = pd.DataFrame({'importances_mean': forest_importances.mean(0), 'importances_std': forest_importances.std(0)}, index=xcols)
    forest_importances_df.sort_values('importances_mean', inplace = True, ascending = False)
    return(forest_importances_df, [np.mean(auc_list), np.mean(acc_list), np.mean(f1_list), np.mean(fm_list), np.std(auc_list), np.std(acc_list), np.std(f1_list), np.std(fm_list)])

def variable_selection(wsi_df, xcols, dtfe, edge_range, nnodes, nreps, nvars, model = 'rf'):
    xcols_reduced = xcols[:]
    for n_ind in [10] + list(range(6, nvars - 1, -1)) + [0]:
        X, y = get_oa_ra_classification_data(wsi_df, xcols_reduced, dtfe, edge_range, nnodes)
        forest_importances_df, score_vals = perform_test(X, y, nreps, xcols_reduced, model = model)
        if forest_importances_df.shape[0] > n_ind:
            xcols_reduced = list(forest_importances_df.index)[:n_ind]
    
    return(forest_importances_df, score_vals)

def get_inclusion_list(inclusion_type):
    if inclusion_type == 'RAvsOA':
        ext_path = 'data/external/cleaned_oa_ra_data_mitre4.csv'
        ext_df = pd.read_csv(ext_path, dtype = object)
        return(list(ext_df['map_id']))
    elif inclusion_type == 'Inflammation':
        with open('data/external/HSS_RA_knee_patients.txt', 'r') as f1:
            ra_knee_ids = [i.replace('\n', '') for i in f1.readlines()]
        return(ra_knee_ids)

xcols = ['bc_frac_dim_wsi_ratio', #'bc_frac_dim_dtfe_wsi', 'bc_frac_dim_full_wsi', 
        'norm_bc_frac_dim_dtfe_wsi', 'num_pixels_total',
        'num_nuclei_total', #'num_nuclei_dtfe_only', #'avg_tissue_per_tile',
        # 'num_clusters', 
        # 'nuclei_density', 
        # 'nuclei_density_dtfe',
        'nuclei_density_aggregates', 'convex_area', 'concave_area', 'eccentricity', 'num_pixels_cluster',
        'num_nuclei_cluster', 'major_axis_length', 'minor_axis_length',
        'bc_frac_dim_cluster', 'norm_pixels_max_cluster',
        'norm_nuclei_max_cluster', 'norm_num_clusters']

nreps = 100
nvars = 3

feature_selection_df = pd.DataFrame(columns = ['dtfe', 'edge_range', 'nnodes',
                                               'feature_1', 'feature_2', 'feature_3',
                                               'feature_1_avg_importance', 'feature_2_avg_importance', 'feature_3_avg_importance', 
                                               'feature_1_importance_std_dev', 'feature_2_importance_std_dev', 'feature_3_importance_std_dev', 
                                               'avg_auc', 'avg_acc', 'avg_f1', 'avg_fm',
                                               'std_auc', 'std_acc', 'std_f1', 'std_fm'])

inclusion_type = 'RAvsOA'
model = 'rf'
proj_list = ['HSS_RA', 'HSS_OA']
inclusion_list = get_inclusion_list(inclusion_type)
wsi_df = get_wsi_df(proj_list, inclusion_list)

start_time = time.time()
#for dtfe in np.round(np.unique(wsi_df.dtfe_threshold), 4):
for dtfe in [0.0001, 0.001]:
    for edge_range in np.unique(wsi_df.edge_range):
        for nnodes in np.unique(wsi_df.number_nodes_necessary_for_cluster):
            forest_importances_df, score_vals = variable_selection(wsi_df, xcols, dtfe, edge_range, nnodes, nreps, nvars)
            new_row = [dtfe, edge_range, nnodes]
            new_row += list(forest_importances_df.index)
            new_row += list(forest_importances_df.importances_mean)
            new_row += list(forest_importances_df.importances_std)            
            new_row += score_vals
            feature_selection_df.loc[feature_selection_df.shape[0]] = new_row
            print(f"{dtfe:.4f}, {edge_range}, {nnodes}: {time.time() - start_time:.0f} seconds")
            print(new_row)

feature_selection_df.to_csv(f'3_feature_model_comparisons_no_nuclei_density_no_num_clusters_{inclusion_type}_{model}.csv', index = False)
# feature_selection_df.to_csv('3_feature_model_comparisons_cleaned_3.csv', index = False)
print('----------')
print(time.time() - start_time)

###############################################################################

def test_model(X, y, nreps, xcols, model = 'rf'):
    auc_list = []
    acc_list = []
    f1_list = []
    fm_list = []
    fpr_list = []
    tpr_list = []
    for i in range(nreps):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        
        if model == 'rf':
            model_object = RandomForestClassifier(n_estimators = 10, max_depth = None, min_samples_split = 2)
        elif model == 'lda':
            model_object = LinearDiscriminantAnalysis()
        elif model == 'logit':
            model_object = LogisticRegression()
        
        model_object.fit(X_train, y_train)
        
        y_score = model_object.predict_proba(X_test)[:,1]
        y_score_labels = model_object.predict(X_test)
        
        f1score = metrics.f1_score(y_test, y_score_labels)
        fmscore = metrics.fowlkes_mallows_score(y_test, y_score_labels)
        
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        auc_list += [roc_auc]
        acc_list += [(y_score_labels == y_test).sum() / len(y_test)]
        f1_list += [f1score]
        fm_list += [fmscore]
        fpr_list += [fpr]
        tpr_list += [tpr]
    
    return([np.mean(auc_list), np.mean(acc_list), np.mean(f1_list), np.mean(fm_list), 
            np.std(auc_list), np.std(acc_list), np.std(f1_list), np.std(fm_list)],
           [fpr_list, tpr_list])


dtfe = 0.003
edge_range = 100
nnodes = 10
nreps = 100

# xcols = ['norm_num_clusters', 'bc_frac_dim_dtfe_wsi', 'nuclei_density_aggregates']
#xcols = ['norm_num_clusters', 'nuclei_density_aggregates', 'nuclei_density_dtfe']
xcols = ['norm_num_clusters', 'nuclei_density_aggregates', 'bc_frac_dim_wsi_ratio']
xcols = ['norm_bc_frac_dim_dtfe_wsi', 'bc_frac_dim_wsi_ratio', 'num_nuclei_total']
xcols = ['norm_num_clusters']

np.random.seed(33)
X, y = get_oa_ra_classification_data(wsi_df, xcols, dtfe, edge_range, nnodes)
score_vals, roc_list = test_model(X, y, nreps, xcols, model = 'rf')

xt = np.linspace(0, 1, 101)
yt_list = []
for i in range(len(roc_list[0])):
    fpr, tpr = roc_list[0][i], roc_list[1][i]
    if i == 0:
        plt.figure()
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    
    plt.plot(
        fpr,
        tpr,
        color = "darkorange",
        lw = 0.3,
    )

    yt_list += [np.interp(xt, fpr, tpr)]

yt = np.mean(yt_list, axis = 0)
plt.plot(
    xt,
    yt,
    'k--',
    lw = 1
)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("DTFE = %.4f, # Nodes = %d \nAUC = %0.2f, Acc = %0.2f, F1 = %0.2f, F-M = %0.2f" % tuple([dtfe, nnodes] + score_vals[:4]))
plt.show()

########################################################################################################################
yvar = 'project'
xcols = ['num_clusters', 'norm_num_clusters', 'nuclei_density', 'nuclei_density_aggregates', 'nuclei_density_dtfe', 'bc_frac_dim_wsi_ratio', 'eccentricity']
xcols = ['bc_frac_dim_wsi_ratio', 'bc_frac_dim_dtfe_wsi', 'bc_frac_dim_full_wsi', 
        'norm_bc_frac_dim_dtfe_wsi', 'num_pixels_total',
        'num_nuclei_total', 'num_nuclei_dtfe_only', 'avg_tissue_per_tile',
        'num_clusters', 
        'nuclei_density', 
        'nuclei_density_dtfe',
        'nuclei_density_aggregates', 'convex_area', 'concave_area', 'eccentricity', 'num_pixels_cluster',
        'num_nuclei_cluster', 'major_axis_length', 'minor_axis_length',
        'bc_frac_dim_cluster', 'norm_pixels_max_cluster',
        'norm_nuclei_max_cluster', 'norm_num_clusters']
xcols = ['num_clusters', 'norm_num_clusters']

edge_range = 100
nnodes = 10

xdf = wsi_df[[yvar, 'ID', 'dtfe_threshold', 'edge_range',
              'number_nodes_necessary_for_cluster'] + xcols].groupby([yvar, 'ID', 'dtfe_threshold', 'edge_range',
              'number_nodes_necessary_for_cluster'])[xcols].max().reset_index()

sub_df = xdf[(xdf.edge_range == edge_range) &
             (xdf.number_nodes_necessary_for_cluster == nnodes)]

fin_df = sub_df[[yvar, 'dtfe_threshold'] + xcols]

for xc in xcols:
    # g = sns.boxplot(data = fin_df, x = 'dtfe_threshold', y = xc, hue = yvar)
    g = sns.pointplot(data = fin_df, x = 'dtfe_threshold', 
                      y = xc, hue = yvar, ci = "sd")
    g.axes.set_title(xc)
    plt.show()

########################################################################################################################
yvar = 'project'
xcols = ['num_clusters', 'norm_num_clusters', 'nuclei_density', 'nuclei_density_aggregates', 'nuclei_density_dtfe', 'bc_frac_dim_wsi_ratio', 'eccentricity']
xcols = ['avg_tissue_per_tile', 'total_tissue', 'num_tiles']

dtfe = 0.007
edge_range = 100
nnodes = 10

xdf = wsi_df[[yvar, 'ID', 'dtfe_threshold', 'edge_range',
              'number_nodes_necessary_for_cluster'] + xcols].groupby([yvar, 'ID', 'dtfe_threshold', 'edge_range',
              'number_nodes_necessary_for_cluster'])[xcols].max().reset_index()

sub_df = xdf[(xdf.dtfe_threshold.round(4) == dtfe) &
             (xdf.edge_range == edge_range) &
             (xdf.number_nodes_necessary_for_cluster == nnodes)]

fin_df = sub_df[[yvar] + xcols]
fin_df['One'] = 1

for xc in xcols:
    #g = sns.boxplot(data = fin_df, x = 'One', y = xc, hue = yvar)
    g = sns.kdeplot(data = fin_df, x = xc, hue = yvar)
    g.axes.set_title(xc)
    plt.show()
##############
xcols = [i for i in wsi_df.columns if 'frac' in i]
xdf = wsi_df[[yvar, 'ID', 'dtfe_threshold', 'edge_range',
              'number_nodes_necessary_for_cluster'] + xcols].groupby([yvar, 'ID', 'dtfe_threshold', 'edge_range',
              'number_nodes_necessary_for_cluster'])[xcols].max().reset_index()

xdf[['dtfe_threshold', 'edge_range','number_nodes_necessary_for_cluster'] + xcols]

###############################################################################
patient_map = {'116': ['116_L_1', '116_L_2', '116_R'],
                 '171': ['171_2', '171'],
                 '172': ['172_1', '172_2'],
                 '175': ['175_1', '175_2'],
                 '176': ['176_2', '176'],
                 '178': ['178_1', '178_2'],
                 '179': ['179_2', '179'],
                 '181': ['181_L_2', '181_L', '181_R_1', '181_R_2'],
                 '182': ['182_1', '182_2'],
                 '183': ['183_L_2', '183_L', '183_R_2', '183_R'],
                 '214': ['214_L', '214_R'],
                 '223': ['223_L', '223_R'],
                 '246': ['246_L', '246_R'],
                 '31': ['31_L', '31_R'],
                 '65': ['65_L', '65_R']}

image_map = {'116_L_1': '116',
            '116_L_2': '116',
            '116_R': '116',
            '171_2': '171',
            '171': '171',
            '172_1': '172',
            '172_2': '172',
            '175_1': '175',
            '175_2': '175',
            '176_2': '176',
            '176': '176',
            '178_1': '178',
            '178_2': '178',
            '179_2': '179',
            '179': '179',
            '181_L_2': '181',
            '181_L': '181',
            '181_R_1': '181',
            '181_R_2': '181',
            '182_1': '182',
            '182_2': '182',
            '183_L_2': '183',
            '183_L': '183',
            '183_R_2': '183',
            '183_R': '183',
            '214_L': '214',
            '214_R': '214',
            '223_L': '223',
            '223_R': '223',
            '246_L': '246',
            '246_R': '246',
            '31_L': '31',
            '31_R': '31',
            '65_L': '65',
            '65_R': '65'}

yvar = 'project'
xcols = ['bc_frac_dim_wsi_ratio', 'bc_frac_dim_dtfe_wsi', 'bc_frac_dim_full_wsi', 
        'norm_bc_frac_dim_dtfe_wsi', 'num_pixels_total',
        'num_nuclei_total', 'num_nuclei_dtfe_only', 'avg_tissue_per_tile',
        'num_clusters', 
        'nuclei_density', 
        'nuclei_density_dtfe',
        'nuclei_density_aggregates', 'convex_area', 'concave_area', 'eccentricity', 'num_pixels_cluster',
        'num_nuclei_cluster', 'major_axis_length', 'minor_axis_length',
        'bc_frac_dim_cluster', 'norm_pixels_max_cluster',
        'norm_nuclei_max_cluster', 'norm_num_clusters']

xcols = ['num_clusters', 'norm_num_clusters', 'nuclei_density', 'nuclei_density_aggregates', 'nuclei_density_dtfe', 'bc_frac_dim_wsi_ratio', 'eccentricity']

edge_range = 100
nnodes = 10

inclusion_list = [i + '_s0' for i in image_map.keys()]
wsi_df = get_wsi_df(['HSS_RA'], inclusion_list)
wsi_df['Patient'] = pd.Categorical([image_map[i.replace('_s0', '')] for i in wsi_df['ID']])
wsi_df['Image'] = wsi_df['ID']

xdf = wsi_df[[yvar, 'Image', 'Patient', 'dtfe_threshold', 'edge_range',
              'number_nodes_necessary_for_cluster'] + xcols].groupby([yvar, 'Image', 'Patient', 'dtfe_threshold', 'edge_range',
              'number_nodes_necessary_for_cluster'])[xcols].max().reset_index()

sub_df = xdf[#((xdf.dtfe_threshold.round(4) == 0.0001) | (xdf.dtfe_threshold.round(4) == 0.007)) &
             (xdf.edge_range == edge_range) &
             (xdf.number_nodes_necessary_for_cluster == nnodes)]

sub_df['One'] = 1
for xc in xcols:
    g = sns.catplot(x = "One", y = xc, hue = "Patient", col = "dtfe_threshold", 
                    data = sub_df, kind = "swarm", sharey = True)
    plt.show()

##############################################################################

xcols = ['bc_frac_dim_wsi_ratio', 'bc_frac_dim_dtfe_wsi', 'bc_frac_dim_full_wsi', 
        'norm_bc_frac_dim_dtfe_wsi', 'num_pixels_total',
        'num_nuclei_total', 'num_nuclei_dtfe_only',
        'num_clusters', 
        'nuclei_density', 
        'nuclei_density_dtfe',
        'nuclei_density_aggregates', 'convex_area', 'concave_area', 'eccentricity', 'num_pixels_cluster',
        'num_nuclei_cluster', 'major_axis_length', 'minor_axis_length',
        'bc_frac_dim_cluster', 'norm_pixels_max_cluster',
        'norm_nuclei_max_cluster', 'norm_num_clusters']

tissue_cols = ['avg_tissue_per_tile', 'num_tiles', 'total_tissue']

corr_list = []
for i, xc in enumerate(xcols):
    temp_list = []
    for j, tc in enumerate(tissue_cols):
        temp_list += [np.corrcoef(wsi_df[xc], wsi_df[tc])[1,0]]
    corr_list += [temp_list]

corr_df = pd.DataFrame(corr_list)
corr_df.columns = tissue_cols
corr_df.index = xcols
corr_df = corr_df.round(2)
corr_df.to_csv('C:/Users/dfrezza/Desktop/temp.csv')

corr_array = np.array(corr_list)
fig, ax = plt.subplots(figsize = (6,6))
im = ax.imshow(np.abs(corr_array))
# im = ax.imshow(corr_array)
ax.xaxis.tick_top()
ax.set_yticks(list(range(len(xcols))), xcols)
ax.set_xticks(list(range(len(tissue_cols))), tissue_cols, rotation = 90)
plt.colorbar(im)
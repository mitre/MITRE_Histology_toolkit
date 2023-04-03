import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import json
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
                if (inclusion_list is None) or (fname.split('_dtfe')[0] in inclusion_list):
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

def get_classification_data(wsi_df, yvar, classes, xcols, dtfe, edge_range, nnodes, include_id = False):
    xdf = wsi_df[[yvar, 'ID', 'dtfe_threshold', 'edge_range',
                  'number_nodes_necessary_for_cluster'] + xcols].groupby([yvar, 'ID', 'dtfe_threshold', 'edge_range',
                  'number_nodes_necessary_for_cluster'])[xcols].max().reset_index()
    
    sub_df = xdf[(xdf.dtfe_threshold.round(4) == dtfe) &
                 (xdf.edge_range == edge_range) &
                 (xdf.number_nodes_necessary_for_cluster == nnodes)]
    
    X = np.array(sub_df[xcols])
    X[np.where(np.isnan(X))] = 0
    if yvar == 'project':
        y = label_binarize(np.array(sub_df[yvar]), classes = classes).flatten()
    elif yvar == 'Inflammation':
        y = np.array(sub_df[yvar])
    if include_id:
        return(X, y, np.array(sub_df['ID']))
    else:
        return(X, y)

def get_oa_ra_classification_data(wsi_df, xcols, dtfe, edge_range, nnodes, include_id = False):
    yvar = 'project'
    classes = ['HSS_RA', 'HSS_OA']
    return(get_classification_data(wsi_df, yvar, classes, xcols, dtfe, edge_range, nnodes, include_id = include_id))

def get_inflam_classification_data(wsi_df, xcols, dtfe, edge_range, nnodes, include_id = False):
    yvar = 'Inflammation'
    classes = []
    return(get_classification_data(wsi_df, yvar, classes, xcols, dtfe, edge_range, nnodes, include_id = include_id))

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
        
        if len(set(y_test)) > 2:
            roc_auc, acc_score, f1score, fmscore = get_scores_multiclass(model_object, X_test, y_test, avg = 'macro', comp = 'ovo')
        else:
            roc_auc, acc_score, f1score, fmscore = get_scores_binary(model_object, X_test, y_test)
        
        auc_list += [roc_auc]
        acc_list += [acc_score]
        f1_list += [f1score]
        fm_list += [fmscore]
    
    forest_importances_df = pd.DataFrame({'importances_mean': forest_importances.mean(0), 'importances_std': forest_importances.std(0)}, index=xcols)
    forest_importances_df.sort_values('importances_mean', inplace = True, ascending = False)
    return(forest_importances_df, [np.mean(auc_list), np.mean(acc_list), np.mean(f1_list), np.mean(fm_list), np.std(auc_list), np.std(acc_list), np.std(f1_list), np.std(fm_list)])

def get_scores_binary(model_object, X_test, y_test):
    y_score = model_object.predict_proba(X_test)[:,1]
    y_score_labels = model_object.predict(X_test)
    
    f1score = metrics.f1_score(y_test, y_score_labels)
    fmscore = metrics.fowlkes_mallows_score(y_test, y_score_labels)
    
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    acc_score = (y_score_labels == y_test).sum() / len(y_test)
    return(roc_auc, acc_score, f1score, fmscore)

def get_scores_multiclass(model_object, X_test, y_test, avg = 'macro', comp = 'ovo'):
    y_score = model_object.predict_proba(X_test)
    y_score_labels = model_object.predict(X_test)
    
    f1score = metrics.f1_score(y_test, y_score_labels, average = avg)
    fmscore = metrics.fowlkes_mallows_score(y_test, y_score_labels)
    
    roc_auc = metrics.roc_auc_score(y_test, y_score, average = avg, multi_class = comp)
    acc_score = (y_score_labels == y_test).sum() / len(y_test)
    return(roc_auc, acc_score, f1score, fmscore)
    
def variable_selection(wsi_df, xcols, dtfe, edge_range, nnodes, nreps, nvars, inclusion_type, model = 'rf'):
    xcols_reduced = xcols[:]
    for n_ind in [10] + list(range(6, nvars - 1, -1)) + [0]:
        if inclusion_type == 'RAvsOA':
            X, y = get_oa_ra_classification_data(wsi_df, xcols_reduced, dtfe, edge_range, nnodes)
        elif inclusion_type == 'Inflammation':
            X, y = get_inflam_classification_data(wsi_df, xcols_reduced, dtfe, edge_range, nnodes)
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
        return(None)
    else:
        with open('data/external/HSS_RA_knee_patients.txt', 'r') as f1:
            ra_knee_ids = [i.replace('\n', '') for i in f1.readlines()]
        return(ra_knee_ids)

def get_confusion(y_true, y_pred):
    y_map_f = {}
    y_map_b = {}
    vals = ['0. None', '1. Mild', '2. Moderate', '3. Marked', '4. Band-like']
    for ii in vals:
        y_map_f[ii] = int(ii[0])
        y_map_b[int(ii[0])] = ii
    
    posibility_map = {}
    for ii in vals:
        intval = int(ii[0])
        posibility_map[ii] = []
        for intval_i in [intval - 1, intval, intval + 1]:
            if intval_i in y_map_b:
                posibility_map[ii] += [y_map_b[intval_i]]
    
    arr = np.zeros((len(vals), 2))
    for ii, jj in enumerate(y_true):
        intval = int(jj[0])
        if y_pred[ii] in posibility_map[jj]:
            arr[intval, 0] += 1
        else:
            arr[intval, 1] += 1
    
    return(np.c_[arr, np.round((arr[:,0] / np.sum(arr, 1)).reshape((len(arr),1)), 2)])

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

nreps = 100
nvars = 3

feature_selection_df = pd.DataFrame(columns = ['dtfe', 'edge_range', 'nnodes',
                                               'feature_1', 'feature_2', 'feature_3',
                                               'feature_1_avg_importance', 'feature_2_avg_importance', 'feature_3_avg_importance', 
                                               'feature_1_importance_std_dev', 'feature_2_importance_std_dev', 'feature_3_importance_std_dev', 
                                               'avg_auc', 'avg_acc', 'avg_f1', 'avg_fm',
                                               'std_auc', 'std_acc', 'std_f1', 'std_fm'])

# inclusion_type = 'RAvsOA'
inclusion_type = 'Inflammation'
# proj_list = ['HSS_RA', 'HSS_OA']
proj_list = ['HSS_RA']
model = 'rf'
inclusion_list = get_inclusion_list(inclusion_type)
wsi_df = get_wsi_df(proj_list, inclusion_list)
with open('data/external/hss_ra_inflammation_map.json', 'r') as json_file:
    # inflam_map = json.load(json_file)
    alt_inflam_map = json.load(json_file)

alt_map_map = {'0': '0', '1': '0', '2': '0', '3': '1', '4': '1'}
alt_map_map = {'0': 0, '1': 0, '2': 0, '3': 1, '4': 1}
alt_map_map = {'0': '0', '1': '0', '2': '1', '3': '2', '4': '2'}
inflam_map = {}
for i in alt_inflam_map:
    if type(alt_inflam_map[i]) is not str:
        inflam_map[i] = np.nan
    else:
        inflam_map[i] = alt_map_map[alt_inflam_map[i][0]]

wsi_df['Inflammation'] = [inflam_map[i] for i in wsi_df['ID']]
wsi_df = wsi_df[wsi_df['Inflammation'].notna()].reset_index(drop = True)

start_time = time.time()
for dtfe in np.round(np.unique(wsi_df.dtfe_threshold), 4):
    for edge_range in np.unique(wsi_df.edge_range):
        for nnodes in np.unique(wsi_df.number_nodes_necessary_for_cluster):
            forest_importances_df, score_vals = variable_selection(wsi_df, xcols, dtfe, edge_range, nnodes, nreps, nvars, inclusion_type, model = model)
            new_row = [dtfe, edge_range, nnodes]
            new_row += list(forest_importances_df.index)
            new_row += list(forest_importances_df.importances_mean)
            new_row += list(forest_importances_df.importances_std)            
            new_row += score_vals
            feature_selection_df.loc[feature_selection_df.shape[0]] = new_row
            print(f"{dtfe:.4f}, {edge_range}, {nnodes}: {time.time() - start_time:.0f} seconds")
            print(new_row)

feature_selection_df.to_csv(f'3_feature_model_comparisons_binary_{inclusion_type}_{model}.csv', index = False)
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2)
        
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


dtfe = 0.007
edge_range = 100
nnodes = 50
nreps = 100

# xcols = ['norm_num_clusters', 'bc_frac_dim_dtfe_wsi', 'nuclei_density_aggregates']
#xcols = ['norm_num_clusters', 'nuclei_density_aggregates', 'nuclei_density_dtfe']
xcols = ['norm_num_clusters', 'nuclei_density_aggregates', 'bc_frac_dim_wsi_ratio']
xcols = ['num_nuclei_total', 'norm_num_clusters', 'nuclei_density_aggregates']
xcols = ['num_nuclei_total', 'nuclei_density', 'nuclei_density_aggregates']
xcols = ['num_nuclei_total', 'nuclei_density']
xcols = ['num_nuclei_total']

np.random.seed(33)
X, y = get_inflam_classification_data(wsi_df, xcols, dtfe, edge_range, nnodes)
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

########################################################################################
def perform_confusion_test(X, y, nreps, xcols, model = 'rf', sample_weight_bool = False):
    auc_list = []
    acc_list = []
    f1_list = []
    fm_list = []
    confusion_list = []
    if sample_weight_bool:
        samp_weight_dict = {str(i0): len(y)/i for i0, i in enumerate(np.unique(y, return_counts = True)[1])}
    for i in range(nreps):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, train_size = 0.85)
        
        if sample_weight_bool:
            samp_weights = [samp_weight_dict[sw] for sw in y_train]
        else:
            samp_weights = None
        
        if model == 'rf':
            model_object = RandomForestClassifier(n_estimators = 10, max_depth = None, min_samples_split = 2, class_weight = 'balanced')
        elif model == 'lda':
            model_object = LinearDiscriminantAnalysis()
        elif model == 'logit':
            model_object = LogisticRegression()
        
        model_object.fit(X_train, y_train, sample_weight = samp_weights)
        
        if len(set(y_test)) > 2:
            roc_auc, acc_score, f1score, fmscore = get_scores_multiclass(model_object, X_test, y_test, avg = 'macro', comp = 'ovo')
        else:
            roc_auc, acc_score, f1score, fmscore = get_scores_binary(model_object, X_test, y_test)
        
        y_score_labels = model_object.predict(X_test)
        auc_list += [roc_auc]
        acc_list += [acc_score]
        f1_list += [f1score]
        fm_list += [fmscore]
        confusion_list += [metrics.confusion_matrix(y_test, y_score_labels, normalize = 'true')]
    
    return(confusion_list, [np.mean(auc_list), np.mean(acc_list), np.mean(f1_list), np.mean(fm_list), np.std(auc_list), np.std(acc_list), np.std(f1_list), np.std(fm_list)])

vals = ['0. None', '1. Mild', '2. Moderate', '3. Marked', '4. Band-like']
vals = ['0. Low', '1. Moderate', '2. High']
dtfe = 0.007
edge_range = 100
nnodes = 50
model = 'rf'
nreps = 100

# xcols = ['norm_num_clusters', 'nuclei_density_aggregates', 'bc_frac_dim_wsi_ratio']
xcols = ['num_nuclei_total', 'norm_num_clusters', 'nuclei_density_aggregates']
# xcols = ['num_nuclei_total', 'nuclei_density', 'nuclei_density_aggregates']
# xcols = ['num_nuclei_total', 'nuclei_density']
# xcols = ['num_nuclei_total']
xcols = ['num_nuclei_total', 'concave_area', 'nuclei_density_aggregates','bc_frac_dim_full_wsi', 'minor_axis_length']

np.random.seed(33)
X, y = get_inflam_classification_data(wsi_df, xcols, dtfe, edge_range, nnodes)
confusion_list, score_vals = perform_confusion_test(X, y, nreps, xcols, model = 'rf', sample_weight_bool = False)

conf_df = pd.DataFrame(np.round(np.array(confusion_list).mean(0), 2))
conf_df.columns = vals
conf_df.index = vals
conf_df.to_csv('temp_conf_mat_out.csv')

########################################################################################

dtfe = 0.007
edge_range = 100
nnodes = 50
model = 'rf'

xcols_reduced = xcols[:]
for n_ind in [10, 5, 0]:
    X, y = get_inflam_classification_data(wsi_df, xcols_reduced, dtfe, edge_range, nnodes)
    np.random.seed(34)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.15)

    if model == 'rf':
        model_object = RandomForestClassifier(n_estimators = 10, max_depth = None, min_samples_split = 2, class_weight = 'balanced')
    elif model == 'lda':
        model_object = LinearDiscriminantAnalysis()
    elif model == 'logit':
        model_object = LogisticRegression()
    
    model_object.fit(X_train, y_train)

    result = permutation_importance(
        model_object, X_test, y_test, n_repeats=10, n_jobs=2
    )
    forest_importances = result['importances'].T
    forest_importances_df = pd.DataFrame({'importances_mean': forest_importances.mean(0), 'importances_std': forest_importances.std(0)}, index=xcols_reduced)
    forest_importances_df.sort_values('importances_mean', inplace = True, ascending = False)

    if forest_importances_df.shape[0] > n_ind:
        xcols_reduced = list(forest_importances_df.index)[:n_ind]


y_score = model_object.predict_proba(X_test)
y_score_labels = model_object.predict(X_test)

print(f'F1-score: \t{metrics.f1_score(y_test, y_score_labels, average = "micro"):.2f}')
print(f'FM-score: \t{metrics.fowlkes_mallows_score(y_test, y_score_labels):.2f}')
print(f'AUC: \t\t{metrics.roc_auc_score(y_test, y_score, average = "macro", multi_class = "ovo"):.2f}')
print('Confusion')
print(metrics.confusion_matrix(y_test, y_score_labels, normalize = 'true'))

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

edge_range = 100
nnodes = 10

xdf = wsi_df[[yvar, 'ID', 'dtfe_threshold', 'edge_range',
              'number_nodes_necessary_for_cluster'] + xcols].groupby([yvar, 'ID', 'dtfe_threshold', 'edge_range',
              'number_nodes_necessary_for_cluster'])[xcols].max().reset_index()

sub_df = xdf[(xdf.edge_range == edge_range) &
             (xdf.number_nodes_necessary_for_cluster == nnodes)]

fin_df = sub_df[[yvar, 'dtfe_threshold'] + xcols]

for xc in xcols:
    g = sns.boxplot(data = fin_df, x = 'dtfe_threshold', y = xc, hue = yvar)
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
from autogluon.tabular import TabularPredictor
from sklearn import model_selection
import skimage.io

dtfe = 0.007
edge_range = 100
nnodes = 50
train_size = 0.8
n_splits = 5
xcols = ['num_nuclei_total', 'norm_num_clusters', 'nuclei_density_aggregates']

X, y, id_series = get_inflam_classification_data(wsi_df, xcols, dtfe, edge_range, nnodes, include_id = True)
np.random.seed(33)
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, id_series, stratify = y, train_size = train_size)
train_data = pd.DataFrame(X_train, columns = xcols)
train_data['outcome'] = y_train
test_data = pd.DataFrame(X_test, columns = xcols)
test_data['outcome'] = y_test

# metric = 'roc_auc'  # specify your evaluation metric here
# predictor = TabularPredictor('outcome', eval_metric=metric,).fit(train_data,num_bag_folds=5, num_bag_sets=1)#, time_limit=time_limit, presets='best_quality')
# predictor.leaderboard(test_data, extra_metrics=['accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted', 'roc_auc', 'roc_auc_ovo_macro', 'average_precision', 'precision'],silent=True)

skf = model_selection.StratifiedKFold(n_splits = n_splits)
group_list = np.zeros(X.shape[0]).astype(int)
i = 0
for train, test in skf.split(X, y):
    group_list[test] = i
    i += 1

all_data = pd.DataFrame(X, columns = xcols)
all_data['outcome'] = y
all_data['cv_groups'] = group_list
# metric = 'accuracy'
# metric = 'balanced_accuracy'
metric = 'roc_auc_ovo_macro'
predictor = TabularPredictor('outcome', problem_type = 'multiclass',
                             eval_metric = metric, groups = 'cv_groups'
                             ).fit(all_data)#, num_bag_folds = 5, num_bag_sets = 1)
print(predictor.leaderboard(silent=True))
print(predictor.leaderboard(all_data, extra_metrics=['accuracy', 'balanced_accuracy', 'f1_macro', 'roc_auc_ovo_macro'],silent=True))


predictor = TabularPredictor('outcome', eval_metric = 'balanced_accuracy').fit(train_data, presets='best_quality')
print(predictor.leaderboard(test_data, extra_metrics=['accuracy', 'balanced_accuracy', 'f1_macro', 'roc_auc_ovo_macro'],silent=True))
model_str = predictor.leaderboard(test_data, extra_metrics=['accuracy', 'balanced_accuracy', 'f1_macro', 'roc_auc_ovo_macro'],silent=True)['model'][0]
pred_df = predictor.predict_proba(test_data, model = model_str)
pred_df['pred'] = predictor.predict(test_data, model = model_str)
pred_df['true'] = test_data['outcome']
pred_df['ID'] = id_test
print(f'Accuracy: {(pred_df["pred"] == pred_df["true"]).sum() / pred_df.shape[0]}')

marked_as_mild = [1, 11, 24, 28]
none_as_marked = [27]
pred_df.iloc[marked_as_mild]
pred_df.iloc[none_as_marked]

clst_dir_name = 'data/processed/clusters/HSS_RA/multiple_params/convex'
nuc_dir_name = 'data/processed/nuclei/HSS_RA'
tile_dir_name = 'data/processed/tiles/HSS_RA'
dtfe_dir_name = 'data/processed/node_features/dtfe/base/HSS_RA'
for idx in marked_as_mild + none_as_marked:
    fname = pred_df['ID'][idx]
    xcls = pd.read_csv(f'{clst_dir_name}/{fname}_dtfe_cluster_key.csv')
    xcls = xcls[(np.round(xcls['threshold'], 4) == dtfe) &
                (xcls['edge_range'] == edge_range) &
                (xcls['number_nodes_necessary_for_cluster'] == nnodes)]
    
    xcls_row = pd.read_csv(f'{clst_dir_name}/{fname}_dtfe_clusters.csv')
    xcls_row = xcls_row[(np.round(xcls_row['threshold'], 4) == dtfe) &
                (xcls_row['edge_range'] == edge_range) &
                (xcls_row['number_nodes_necessary_for_cluster'] == nnodes)]
    
    xcls = xcls.round(1).merge(xcls_row[['cluster_ID', 'centroid_x', 'centroid_y']].round(1), how = 'inner', left_on = ['cluster_dtfe_x', 'cluster_dtfe_y'], right_on = ['centroid_x', 'centroid_y'])[['cluster_ID', 'nuclei_x_wsi', 'nuclei_y_wsi']]
    cluster_counts = np.unique(xcls.cluster_ID, return_counts = True)
    cluster_id = cluster_counts[0][np.argmax(cluster_counts[1])]
    
    xnuc = pd.read_csv(f'{nuc_dir_name}/{fname}_nuclei_pos.csv')
    xsingle = xcls[xcls['cluster_ID'] == cluster_id].merge(xnuc[['row', 'col', 'nuclei_x_tile', 'nuclei_y_tile', 'nuclei_x_wsi', 'nuclei_y_wsi']].round(1), how = 'inner')
    row_cols = xsingle[['row','col']].drop_duplicates().values
    tile_array = np.empty((np.unique(row_cols[:,0]).shape[0] * 1024, np.unique(row_cols[:,1]).shape[0] * 1024, 3)).astype(int)
    min_row, min_col = row_cols.min(axis = 0)
    x_pos = []
    y_pos = []
    for row, col in row_cols:
        npz_fname = f'{tile_dir_name}/{fname}/{fname}__R{row}_C{col}_TS1024_OL0.jpg'
        start_row = (row - min_row) * 1024
        end_row = (row + 1 - min_row) * 1024
        start_col = (col - min_col) * 1024
        end_col = (col + 1 - min_col) * 1024        
        tile_array[start_row:end_row, start_col:end_col] = skimage.io.imread(npz_fname)
        x_pos += (xnuc[(xnuc['row'] == row) & (xnuc['col'] == col)]['nuclei_x_tile'] + 1024 * (col - min_col)).to_list()
        y_pos += (xnuc[(xnuc['row'] == row) & (xnuc['col'] == col)]['nuclei_y_tile'] + 1024 * (row - min_row)).to_list()
    
    plt.figure(figsize = (12,12))
    plt.imshow(tile_array)
    plt.scatter(x_pos, y_pos, s = 50, c = 'cyan', marker = 'p')

    xdtfe = pd.read_csv(f'{dtfe_dir_name}/{fname}_base_dtfe.csv')
    
    tile_idx = np.where((xnuc['row'] > 12) & (xnuc['row'] < 16) &
                        (xnuc['col'] > 12) & (xnuc['col'] < 16))[0]
    fig, ax = plt.subplots(ncols = 2, figsize = (10,5))
    ax[0].scatter(xdtfe['nuclei_x_wsi'], xdtfe['nuclei_y_wsi'], c = xdtfe['dtfe'], s = 1, cmap = 'jet', vmax = 0.005)#dtfe)
    ax[1].scatter(xnuc['nuclei_x_wsi'], xnuc['nuclei_y_wsi'], s = 1)
    ax[1].scatter(xcls['nuclei_x_wsi'], xcls['nuclei_y_wsi'], s = 1)
    #ax[0].scatter(xnuc['nuclei_x_wsi'][tile_idx], xnuc['nuclei_y_wsi'][tile_idx], s = 1)
    fig.suptitle(f'Patient {fname.split("_s0")[0]}\nGrade: {pred_df["true"][idx]}, Assigned To: {pred_df["pred"][idx]}', fontsize = 15)
    ax[0].set_aspect(1)
    ax[1].set_aspect(1)
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    
##############################################################################

model_str = 'RandomForestEntr_BAG_L1'
task = TabularPredictor('outcome', groups = 'cv_groups')
task.load(path = 'AutogluonModels/ag-20220214_185802')
predictor_cv = task.fit(
        train_data = all_data,#train_data,
        auto_stack = True
)

predictor_info = predictor_cv._learner.get_info(include_model_info=True)
model_info = predictor_info['model_info'][model_str]
cv_scores = [child_info['val_score'] for child, child_info in model_info['children_info'].items()]
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

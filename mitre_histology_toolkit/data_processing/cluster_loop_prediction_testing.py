import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
pd.set_option('display.max_columns', None)

# read in knee patient ids to limit RA samples
with open('data/external/HSS_RA_knee_patients.txt', 'r') as f1:
    ra_knee_ids = [i.replace('\n', '') for i in f1.readlines()]

ext_path = 'data/external/cleaned_oa_ra_data_mitre4.csv'
ext_df = pd.read_csv(ext_path, dtype = object)
inclusion_list = list(ext_df['map_id'])

proj_list = ['HSS_RA', 'HSS_OA']
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

import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import time

xcols = ['bc_frac_dim_dtfe_wsi', 'bc_frac_dim_full_wsi', 'bc_frac_dim_wsi_ratio',
        'norm_bc_frac_dim_dtfe_wsi', 'num_clusters', 'num_pixels_total',
        'num_nuclei_total', 'num_nuclei_dtfe_only', 'avg_tissue_per_tile',
        'num_tiles', 'original_num_nuclei', 'nuclei_density', 'nuclei_density_dtfe',
        'nuclei_density_aggregates', 'convex_area', 'concave_area', 'eccentricity', 'num_pixels_cluster',
        'num_nuclei_cluster', 'major_axis_length', 'minor_axis_length',
        'bc_frac_dim_cluster', 'total_tissue', 'norm_pixels_max_cluster',
        'norm_nuclei_max_cluster', 'norm_num_clusters']

xcols = ['bc_frac_dim_dtfe_wsi', 'bc_frac_dim_full_wsi', 'bc_frac_dim_wsi_ratio',
        'norm_bc_frac_dim_dtfe_wsi', 'num_clusters', 'num_pixels_total',
        'num_nuclei_total', 'num_nuclei_dtfe_only', 'avg_tissue_per_tile',
        'nuclei_density', 'nuclei_density_dtfe',
        'nuclei_density_aggregates', 'convex_area', 'concave_area', 'eccentricity', 'num_pixels_cluster',
        'num_nuclei_cluster', 'major_axis_length', 'minor_axis_length',
        'bc_frac_dim_cluster', 'norm_pixels_max_cluster',
        'norm_nuclei_max_cluster', 'norm_num_clusters']

xcols = ['norm_pixels_max_cluster',
        'norm_nuclei_max_cluster', 'norm_num_clusters']

xcols = ['bc_frac_dim_dtfe_wsi',
        'bc_frac_dim_full_wsi', 'num_pixels_total',
        'num_nuclei_total', 'num_nuclei_dtfe_only', 'avg_tissue_per_tile',
        'original_num_nuclei',
        'eccentricity',
        'minor_axis_length',
        'norm_nuclei_max_cluster', 'norm_num_clusters']

# this is best for 009 and 100
xcols = ['norm_bc_frac_dim_dtfe_wsi',
        'avg_tissue_per_tile',
        'nuclei_density_aggregates']
xcols = ['avg_tissue_per_tile']

# this is best for 007 and 100
xcols = ['norm_bc_frac_dim_dtfe_wsi',
        'avg_tissue_per_tile',
        'nuclei_density_aggregates']

# this is best for 005 and 50
xcols = ['norm_bc_frac_dim_dtfe_wsi',
        'avg_tissue_per_tile',
        'eccentricity']

# this is best for 003 and 10
xcols = ['norm_bc_frac_dim_dtfe_wsi',
        'avg_tissue_per_tile',
        'norm_num_clusters']

xdf = wsi_df[['project', 'ID', 'dtfe_threshold', 'edge_range',
              'number_nodes_necessary_for_cluster'] + xcols].groupby(['project', 'ID', 'dtfe_threshold', 'edge_range',
              'number_nodes_necessary_for_cluster'])[xcols].max().reset_index()

xdict = {}
for dtfe in np.round(np.unique(xdf.dtfe_threshold), 3):
    if dtfe not in xdict:
        xdict[dtfe] = {}
    for edge_range in np.unique(xdf.edge_range):
        if edge_range not in xdict[dtfe]:
            xdict[dtfe][edge_range] = {}
        for nnodes in np.unique(xdf.number_nodes_necessary_for_cluster):
            if nnodes not in xdict[dtfe][edge_range]:
                xdict[dtfe][edge_range][nnodes] = {'X': {}, 'y': {}}
                
            
            sub_df = xdf[(xdf.dtfe_threshold.round(3) == dtfe) &
                         (xdf.edge_range == edge_range) &
                         (xdf.number_nodes_necessary_for_cluster == nnodes)]
            
            xdict[dtfe][edge_range][nnodes]['X'] = np.array(sub_df[xcols])
            xdict[dtfe][edge_range][nnodes]['y'] = np.array(sub_df[['project']])

dtfe = 0.003
edge_range = 100
nnodes = 10
X = xdict[dtfe][edge_range][nnodes]['X'].copy()
X[np.where(np.isnan(X))] = 0
y = label_binarize(xdict[dtfe][edge_range][nnodes]['y'], classes = ['HSS_RA', 'HSS_OA']).flatten()

# scorings = ['accuracy', 'roc_auc_ovr', 'roc_auc_ovo', 
#             'f1_macro', 'fowlkes_mallows_score',
#             'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']

# score_dict = {}
# index_list = []
# for scoring in scorings:
#     score_dict[scoring] = []
#     for cv_int in [2, 3, 5, 10, 20]:
#         clf = RandomForestClassifier(n_estimators = 10, max_depth = None, min_samples_split = 2)
#         scores = cross_val_score(clf, X, y, cv = cv_int, scoring = scoring)
#         score_dict[scoring] += [scores.min(), scores.mean(), scores.max()]
#         for agg in ['min', 'mean', 'max']:
#             index_list += [f'{cv_int} folds: {agg}']

# score_df = pd.DataFrame(score_dict)
# score_df.index = index_list[:score_df.shape[0]]

# print(score_df)
# score_df.round(2).to_csv(f'score_testing_{dtfe}_{nnodes}.csv')
auc_list = []
f1_list = []
fm_list = []
for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    forest = RandomForestClassifier(n_estimators = 10, max_depth = None, min_samples_split = 2)
    forest.fit(X_train, y_train)
    
    start_time = time.time()
    result = permutation_importance(
        forest, X_test, y_test, n_repeats=10, n_jobs=2
    )
    elapsed_time = time.time() - start_time
    print(f"{i+1}. Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    
    if i == 0:
        forest_importances = result['importances'].T
    else:
        forest_importances = np.r_[forest_importances, result['importances'].T]
    # del result['importances']
    # forest_importances = pd.DataFrame(result, index=xcols)
    # forest_importances.sort_values('importances_mean', inplace = True)
    
    # fig, ax = plt.subplots(figsize = (8,8))
    # forest_importances.importances_mean.plot.bar(yerr=forest_importances.importances_std, ax=ax)
    # ax.set_title("Feature importances using permutation on full model")
    # ax.set_ylabel("Mean accuracy decrease")
    # fig.tight_layout()
    # plt.show()
    
    
    y_score = forest.predict_proba(X_test)[:,1]
    y_score_labels = forest.predict(X_test)
    
    f1score = metrics.f1_score(y_test, y_score_labels)
    fmscore = metrics.fowlkes_mallows_score(y_test, y_score_labels)
    
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    auc_list += [roc_auc]
    f1_list += [f1score]
    fm_list += [fmscore]
    
    if i == 0:
        plt.figure()
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw = 0.3,
    )

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("DTFE = %.3f, # Nodes = %d, # Features = %d\nAUC = %0.2f, F1 = %0.2f, F-M = %0.2f" % (dtfe, nnodes, forest_importances.shape[1],np.mean(auc_list), np.mean(f1_list), np.mean(fm_list)))
plt.show()

forest_importances_df = pd.DataFrame({'importances_mean': forest_importances.mean(0), 'importances_std': forest_importances.std(0)}, index=xcols)
forest_importances_df.sort_values('importances_mean', inplace = True)

fig, ax = plt.subplots(figsize = (8,8))
forest_importances_df.importances_mean.plot.bar(yerr=forest_importances_df.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()






plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw = lw,
    label="AUC = %0.2f\nF1 = %0.2f\nF-M = %0.2f" % (np.mean(auc_list), np.mean(f1_list), np.mean(fm_list)),
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f'DTFE = {dtfe}, # Nodes = {nnodes}, # Features = {forest_importances.shape[1]}')
plt.legend(loc="lower right")
plt.show()



"""
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2)
scores = cross_val_score(clf, X, y, cv=5)
print(scores.mean())

clf = RandomForestClassifier(n_estimators = 10, max_depth = None, min_samples_split = 2)
scores = cross_val_score(clf, X, y, cv = 5, scoring = 'roc_auc_ovo')
print(scores.mean())

clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2)
scores = cross_val_score(clf, X, y, cv=5)
print(scores.mean())
"""

from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

ypca = X.dot(pca.components_[0])




xcols = ['bc_frac_dim_dtfe_wsi',
        'bc_frac_dim_full_wsi', 'num_clusters', 'num_pixels_total',
        'num_nuclei_total', 'num_nuclei_dtfe_only', 'avg_tissue_per_tile',
        'num_tiles', 'original_num_nuclei',
        'convex_area', 'concave_area', 'eccentricity', 'num_pixels_cluster',
        'num_nuclei_cluster', 'major_axis_length', 'minor_axis_length',
        'bc_frac_dim_cluster']

xcols = ['num_clusters', 'num_nuclei_cluster', 'num_pixels_total']
xdf = wsi_df[['project', 'ID', 'dtfe_threshold', 'edge_range',
              'number_nodes_necessary_for_cluster'] + xcols].groupby(['project', 'ID', 'dtfe_threshold', 'edge_range',
              'number_nodes_necessary_for_cluster'])[xcols].max().reset_index()


g = sns.catplot(data = xdf, x = "num_clusters", y = "num_pixels_total", hue = "project",
                row = "dtfe_threshold", col = "number_nodes_necessary_for_cluster")

g.fig.set_figheight(8)
g.fig.set_figwidth(8)

h = sns.scatterplot(data = xdf, x = 'dtfe_threshold', y = 'number_nodes_necessary_for_cluster',
                    hue = 'project')


"""
SCORERS = dict(explained_variance=explained_variance_scorer,
               r2=r2_scorer,
               max_error=max_error_scorer,
               neg_median_absolute_error=neg_median_absolute_error_scorer,
               neg_mean_absolute_error=neg_mean_absolute_error_scorer,
               neg_mean_squared_error=neg_mean_squared_error_scorer,
               neg_mean_squared_log_error=neg_mean_squared_log_error_scorer,
               neg_root_mean_squared_error=neg_root_mean_squared_error_scorer,
               neg_mean_poisson_deviance=neg_mean_poisson_deviance_scorer,
               neg_mean_gamma_deviance=neg_mean_gamma_deviance_scorer,
               accuracy=accuracy_scorer, roc_auc=roc_auc_scorer,
               roc_auc_ovr=roc_auc_ovr_scorer,
               roc_auc_ovo=roc_auc_ovo_scorer,
               roc_auc_ovr_weighted=roc_auc_ovr_weighted_scorer,
               roc_auc_ovo_weighted=roc_auc_ovo_weighted_scorer,
               balanced_accuracy=balanced_accuracy_scorer,
               average_precision=average_precision_scorer,
               neg_log_loss=neg_log_loss_scorer,
               neg_brier_score=neg_brier_score_scorer,
               # Cluster metrics that use supervised evaluation
               adjusted_rand_score=adjusted_rand_scorer,
               homogeneity_score=homogeneity_scorer,
               completeness_score=completeness_scorer,
               v_measure_score=v_measure_scorer,
               mutual_info_score=mutual_info_scorer,
               adjusted_mutual_info_score=adjusted_mutual_info_scorer,
               normalized_mutual_info_score=normalized_mutual_info_scorer,
               fowlkes_mallows_score=fowlkes_mallows_scorer)

"""

def get_data(wsi_df, xcols, dtfe, edge_range, nnodes):
    xdf = wsi_df[['project', 'ID', 'dtfe_threshold', 'edge_range',
                  'number_nodes_necessary_for_cluster'] + xcols].groupby(['project', 'ID', 'dtfe_threshold', 'edge_range',
                  'number_nodes_necessary_for_cluster'])[xcols].max().reset_index()
    
    sub_df = xdf[(xdf.dtfe_threshold.round(3) == dtfe) &
                 (xdf.edge_range == edge_range) &
                 (xdf.number_nodes_necessary_for_cluster == nnodes)]
                
    X = np.array(sub_df[xcols])
    X[np.where(np.isnan(X))] = 0
    y = label_binarize(np.array(sub_df[['project']]), classes = ['HSS_RA', 'HSS_OA']).flatten()
    #print(sub_df[xcols].head())
    return(X, y)
    

def perform_test(X, y, nreps, xcols):
    auc_list = []
    f1_list = []
    fm_list = []
    for i in range(nreps):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y)
        
        forest = RandomForestClassifier(n_estimators = 10, max_depth = None, min_samples_split = 2)
        forest.fit(X_train, y_train)
        
        result = permutation_importance(
            forest, X_test, y_test, n_repeats=10, n_jobs=2
        )
        
        if i == 0:
            forest_importances = result['importances'].T
        else:
            forest_importances = np.r_[forest_importances, result['importances'].T]
        
        y_score = forest.predict_proba(X_test)[:,1]
        y_score_labels = forest.predict(X_test)
        
        f1score = metrics.f1_score(y_test, y_score_labels)
        fmscore = metrics.fowlkes_mallows_score(y_test, y_score_labels)
        
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        auc_list += [roc_auc]
        f1_list += [f1score]
        fm_list += [fmscore]
    
    forest_importances_df = pd.DataFrame({'importances_mean': forest_importances.mean(0), 'importances_std': forest_importances.std(0)}, index=xcols)
    forest_importances_df.sort_values('importances_mean', inplace = True, ascending = False)
    return(forest_importances_df, [np.mean(auc_list), np.mean(f1_list), np.mean(fm_list), np.std(auc_list), np.std(f1_list), np.std(fm_list)])

def variable_selection(wsi_df, xcols, dtfe, edge_range, nnodes, nreps, nvars):
    xcols_reduced = xcols[:]
    for n_ind in [10] + list(range(6, nvars - 1, -1)) + [0]:
        X, y = get_data(wsi_df, xcols_reduced, dtfe, edge_range, nnodes)
        forest_importances_df, score_vals = perform_test(X, y, nreps, xcols_reduced)
        if forest_importances_df.shape[0] > n_ind:
            xcols_reduced = list(forest_importances_df.index)[:n_ind]
    
    return(forest_importances_df, score_vals)

xcols = ['bc_frac_dim_wsi_ratio', #'bc_frac_dim_dtfe_wsi', 'bc_frac_dim_full_wsi', 
        'norm_bc_frac_dim_dtfe_wsi', 'num_clusters', 'num_pixels_total',
        'num_nuclei_total', #'num_nuclei_dtfe_only', #'avg_tissue_per_tile',
        'nuclei_density', 
        #'nuclei_density_dtfe',
        'nuclei_density_aggregates', 'convex_area', 'concave_area', 'eccentricity', 'num_pixels_cluster',
        'num_nuclei_cluster', 'major_axis_length', 'minor_axis_length',
        'bc_frac_dim_cluster', 'norm_pixels_max_cluster',
        'norm_nuclei_max_cluster', 'norm_num_clusters']


# xcols_reduced = ['norm_num_clusters', 'bc_frac_dim_dtfe_wsi', 'norm_bc_frac_dim_dtfe_wsi']
# xcols_reduced = ['bc_frac_dim_dtfe_wsi','norm_num_clusters','nuclei_density']
nreps = 100
nvars = 3

feature_selection_df = pd.DataFrame(columns = ['dtfe', 'edge_range', 'nnodes',
                                               'feature_1', 'feature_2', 'feature_3',
                                               'feature_1_avg_importance', 'feature_2_avg_importance', 'feature_3_avg_importance', 
                                               'feature_1_importance_std_dev', 'feature_2_importance_std_dev', 'feature_3_importance_std_dev', 
                                               'avg_auc', 'avg_f1', 'avg_fm',
                                               'std_auc', 'std_f1', 'std_fm'])

start_time = time.time()
for dtfe in np.round(np.unique(wsi_df.dtfe_threshold), 3):
    for edge_range in np.unique(wsi_df.edge_range):
        for nnodes in np.unique(wsi_df.number_nodes_necessary_for_cluster):
            forest_importances_df, score_vals = variable_selection(wsi_df, xcols, dtfe, edge_range, nnodes, nreps, nvars)
            new_row = [dtfe, edge_range, nnodes]
            new_row += list(forest_importances_df.index)
            new_row += list(forest_importances_df.importances_mean)
            new_row += list(forest_importances_df.importances_std)            
            new_row += score_vals
            feature_selection_df.loc[feature_selection_df.shape[0]] = new_row
            print(f"{dtfe:.3f}, {edge_range}, {nnodes}: {time.time() - start_time:.0f} seconds")
            print(new_row)

# feature_selection_df.to_csv('3_feature_model_comparisons_knee_3_no_nuclei_density.csv', index = False)
feature_selection_df.to_csv('3_feature_model_comparisons_cleaned_3.csv', index = False)


###############################################################################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

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
X, y = get_data(wsi_df, xcols, dtfe, edge_range, nnodes)
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
plt.title("DTFE = %.3f, # Nodes = %d \nAUC = %0.2f, Acc = %0.2f, F1 = %0.2f, F-M = %0.2f" % tuple([dtfe, nnodes] + score_vals[:4]))
plt.show()

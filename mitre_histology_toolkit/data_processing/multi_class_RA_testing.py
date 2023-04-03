import pandas as pd
import numpy as np
import copy

class OrdinalClassifier():

    def __init__(self, clf):
        self.clf = clf
        self.clfs = {}

    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = copy.deepcopy(self.clf)
                clf.fit(X, binary_y)
                self.clfs[i] = clf

    def predict_proba(self, X):
        clfs_predict = {k:self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []
        for i,y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn import metrics

class KNeighborsOrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, *, weights='uniform', 
                 algorithm='auto', leaf_size=30, p=2, 
                 metric='minkowski', metric_params=None, n_jobs=None):
        
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        
        self.clf_ = KNeighborsClassifier(**self.get_params())
        self.clfs_ = {}
        self.classes_ = np.sort(np.unique(y))
        if self.classes_.shape[0] > 2:
            for i in range(self.classes_.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.classes_[i]).astype(np.uint8)
                clf = clone(self.clf_)
                clf.fit(X, binary_y)
                self.clfs_[i] = clf
        return self
    
    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        clfs_predict = {k:self.clfs_[k].predict_proba(X) for k in self.clfs_}
        predicted = []
        for i,y in enumerate(self.classes_):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T
    
    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        return np.argmax(self.predict_proba(X), axis=1)

def get_classification_data(wsi_df, yvar, classes, xcols, dtfe, edge_range, nnodes):
    xdf = wsi_df[[yvar, 'ID', 'dtfe_threshold', 'edge_range',
                  'number_nodes_necessary_for_cluster'] + xcols].groupby([yvar, 'ID', 'dtfe_threshold', 'edge_range',
                  'number_nodes_necessary_for_cluster'])[xcols].max().reset_index()
    
    sub_df = xdf[(xdf.dtfe_threshold.round(4) == dtfe) &
                 (xdf.edge_range == edge_range) &
                 (xdf.number_nodes_necessary_for_cluster == nnodes)]
    
    X = np.array(sub_df[xcols])
    X[np.where(np.isnan(X))] = 0
    y = np.array(sub_df[yvar])
    return(X, y)

inflam_crosswalk = {'None': '0. None', 'Mild (0-1 perivascular aggregates per low power field)': '1. Mild', 'Moderate (>1 perivascular aggregate + focal interstitial infiltration)': '2. Moderate', 'Marked (both perivascular and widespread interstitial aggregates)': '3. Marked', 'Band-like': '4. Band-like'} 
inflam_crosswalk_int = {'None': 0, 'Mild (0-1 perivascular aggregates per low power field)': 1, 'Moderate (>1 perivascular aggregate + focal interstitial infiltration)': 2, 'Marked (both perivascular and widespread interstitial aggregates)': 3, 'Band-like': 4} 

id_map = {}
hss_ra_map = pd.read_csv('data/external/HSS_RA_nuclei_density.csv')
for i in range(hss_ra_map.shape[0]):
    id_map[hss_ra_map['file_name'][i].replace('svs.csv', 's0')] = int(hss_ra_map['subject_id'][i])

inflam = pd.read_csv('data/external/HSSFLARE_PathologyCheck_DATA_LABELS_2020-02-07.csv')
inflam_map = {}
for tkey in id_map:
    if id_map[tkey] in list(inflam['Study ID']):
        inflam_map[tkey] = inflam_crosswalk[inflam[inflam['Study ID'] == id_map[tkey]]['Synovial Lymphocytic Inflammation'].values[0]]

xcols = ['bc_frac_dim_wsi_ratio', #'bc_frac_dim_dtfe_wsi', 'bc_frac_dim_full_wsi', 
        'norm_bc_frac_dim_dtfe_wsi', 'num_pixels_total',
        'num_nuclei_total', #'num_nuclei_dtfe_only', #'avg_tissue_per_tile',
        'num_clusters', 
        'nuclei_density', 
        'nuclei_density_dtfe',
        'nuclei_density_aggregates', 'convex_area', 'concave_area', 'eccentricity', 'num_pixels_cluster',
        'num_nuclei_cluster', 'major_axis_length', 'minor_axis_length',
        'bc_frac_dim_cluster', 'norm_pixels_max_cluster',
        'norm_nuclei_max_cluster', 'norm_num_clusters']

xcols = ['nuclei_density', 'num_pixels_total', 'nuclei_density_dtfe', 'bc_frac_dim_wsi_ratio']

inclusion_type = 'Inflammation'
proj_list = ['HSS_RA']
inclusion_list = get_inclusion_list(inclusion_type)
wsi_df = get_wsi_df(proj_list, inclusion_list)
all_ids = set(wsi_df['ID'])
for i in all_ids:
    if i not in inflam_map:
        inflam_map[i] = np.nan

    
wsi_df['Inflammation'] = [inflam_map[i] for i in wsi_df['ID']]
wsi_df = wsi_df[wsi_df['Inflammation'].notna()].reset_index(drop = True)

dtfe = 0.0001
edge_range = 100
nnodes = 10
yvar = 'Inflammation'
classes = ['HSS_RA']

X, y = get_classification_data(wsi_df, yvar, classes, xcols, dtfe, edge_range, nnodes)
y = np.where(y > 2, 1, 0)
#n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
myclass = KNeighborsOrdinalClassifier()
myclass.fit(X,y)

metrics.confusion_matrix(y, myclass.predict(X))
metrics.confusion_matrix(y, myclass.predict(X), normalize='true')
metrics.confusion_matrix(y, myclass.predict(X), normalize='all')
metrics.confusion_matrix(y, myclass.predict(X), normalize='pred')


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
model = 'rf'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)
# X_train = copy.copy(X)
# y_train = copy.copy(y)
# X_test = X_train
# y_test = y_train

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

forest_importances = result['importances'].T

y_score = model_object.predict_proba(X_test)
y_score_labels = model_object.predict(X_test)

for avg in ['micro', 'macro', 'weighted']:
    print(f'{avg}: {metrics.f1_score(y_test, y_score_labels, average = avg)}')

print(metrics.fowlkes_mallows_score(y_test, y_score_labels))

for avg in ['macro', 'weighted']:
    for comp in ['ovo', 'ovr']:
        # print(roc_auc_score(y_test, y_score[:,0], average = avg, multi_class = comp))
        print(roc_auc_score(y_test, y_score, average = avg, multi_class = comp))


forest_importances_df = pd.DataFrame({'importances_mean': forest_importances.mean(0), 'importances_std': forest_importances.std(0)}, index=xcols)
forest_importances_df.sort_values('importances_mean', inplace = True, ascending = False)


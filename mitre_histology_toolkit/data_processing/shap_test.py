import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import shap
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

# Read the data 
data = pd.read_csv('shap_test.csv')
columns = data.drop(['outcome'], axis = 1).columns
columns = ['num_nuclei_total', 'concave_area', 'nuclei_density_aggregates','bc_frac_dim_full_wsi', 'minor_axis_length']
columns = ['num_nuclei_total', 'norm_num_clusters', 'nuclei_density_aggregates']

# X = pd.get_dummies(data)
# X = np.array(X.drop(['outcome'], axis=1))
X = np.array(data[columns])
y = np.array([str(i) for i in data['outcome']])

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train = pd.DataFrame(X_train, columns=columns)
X_test = pd.DataFrame(X_test, columns=columns)

#training model
clf = RandomForestClassifier(n_estimators = 10, max_depth = None, min_samples_split = 2)
clf.fit(X_train, y_train)

#explaining model
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)
#shap_values of 1 for positive label
shap.summary_plot(shap_values[1], X_test)

#loading and preparing the data

#if you don't shuffle you wont need to keep track of test_index, but I think 
#it is always good practice to shuffle your data
kf = KFold(n_splits=5,shuffle=True)

list_shap_values = list()
list_test_sets = list()
accuracy = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train = pd.DataFrame(X_train,columns=columns)
    X_test = pd.DataFrame(X_test,columns=columns)

    #training model
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)

    #explaining model
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    #for each iteration we save the test_set index and the shap_values
    list_shap_values.append(shap_values)
    list_test_sets.append(test_index)
    accuracy += [(clf.predict(X_test) == y_test).sum() / y_test.shape[0]]


#combining results from all iterations
test_set = list_test_sets[0]
shap_values = np.array(list_shap_values[0])
for i in range(1,len(list_test_sets)):
    test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
    shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=1)
#bringing back variable names    
X_test = pd.DataFrame(X[test_set],columns=columns)

#creating explanation plot for the whole experiment, the first dimension from shap_values indicate the class we are predicting (0=0, 1=1)
for idx in range(len(shap_values)):
    shap.summary_plot(shap_values[idx], X_test)
    plt.show()

#dependence plot, the first number (0) is the index of the column to be plotted
shap.dependence_plot(0,shap_values[1], X_test)

# model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.001)
model = RandomForestClassifier(n_estimators = 10, max_depth = None, min_samples_split = 2)
model.fit(X, y)

# load JS visualization code to notebook
shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# i = 4
# shap.force_plot(explainer.expected_value, shap_values[i], features=X.iloc[i], feature_names=X.columns)
shap.summary_plot(shap_values, features=X_test, feature_names=columns)
shap.summary_plot(shap_values, features=X, feature_names=X.columns, plot_type='bar')


# vals = np.abs(shap_values.values).mean(0)
# feature_names = train_x.columns()

# feature_importance = pd.DataFrame(list(zip(feature_names, vals)),
#                                  columns=['col_name','feature_importance_vals'])
# feature_importance.sort_values(by=['feature_importance_vals'],
#                               ascending=False, inplace=True)
# feature_importance.head()

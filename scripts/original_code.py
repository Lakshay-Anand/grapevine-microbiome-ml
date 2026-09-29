# Original Code
#
# This file keeps the original executable statements used for the analysis and adds
# explanations around them. It  combine code cells from several notebook analyses, so
# the order and the variables left in memory can affect what runs.
#
# Intended high-level workflow:
# 1. Load a processed sample-by-ASV feature table from FeatureDataWoOut.pkl.
# 2. Choose one response label and retain the desired samples/classes.
# 3. Encode labels, split samples, fit classifiers, and evaluate predictions.
# 4. Run a separate neural-network cross-validation search and inspect SHAP values.
#
# Important reading notes:
# - The Approach A and Approach B lists are alternatives, but this file assigns
#   several of them in succession; later assignments replace earlier ones.
# - The feature filters below are also applied in succession to the same column.
# - Comments below describe the code as written and flag mismatches; they do not
#   silently repair the original analysis logic.

#### Importing Libraries

# The imports below provide array/data-frame handling, model training, metrics,
# neural-network layers, and statistics. Several are duplicated from merged
# notebook cells. Some classifiers used later (for example RandomForestClassifier,
# BernoulliNB, GaussianNB, SVC, and KNeighborsClassifier) are not imported here.


import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
import gc
from scipy.stats import sem
import scipy.stats

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
import gc
from scipy.stats import sem
import scipy.stats

# Ask TensorFlow which physical GPU devices it can see. This expression returns
# a list, but the script does not store or print that list explicitly.
tf.config.list_physical_devices('GPU')

'''
We used two different filtering criteria for sparse classes to build the ML models.
Approach A: Inclusion of Sparse Classes: In this approach, any class with a minimum of 3 samples was retained, maximizing the number of classes at the cost of potential over-representation of the majority class(es). 
Approach B: Exclusion of Sparse Classes: Here, only classes with at least 15 samples were retained. This criterion ensured a more balanced representation of classes for robust evaluation at the expense of reduced class diversity. 
'''

# LABELS, CLASS FILTERS, AND INPUT DATA
# These lists are intended to restrict which countries, continents, or grape
# varieties enter an analysis. The two consecutive sets of assignments below
# correspond to different sparse-class approaches; the second set overwrites
# the first set because both use the same variable names.

''' for preventing code erdudancy, variables are set according to the two different labels "class_label" . selected countries, continents, and varieties are defined for both Approach A and Approach B below '''

# Approach B selection values. These replace the Approach A values above; only
# the values in these final assignments remain bound to countries, varieties_kept,
# and conti when the filters later run.


countries = ['USA' ,'Spain' ,'Croatia' ,'France' ,'Hungary' ,'Italy' ,'Argentina','South Africa','Australia', 'Denmark', 'Portugal' ,'Germany']
varieties_kept = ['Merlot', 'Tempranillo' ,'Callet' ,'Cabernet Sauvignon' ,'Verdejo' \
  ,'Sauvignon Blanc' ,'Cabernet Franc'  \
 ,'Chardonnay' ,'Pinot Noir'  ,'Grenache'  \
   , 'Airen' ,'Shiraz'  ,'Barbera' \
  ,'Malbec' ,'Rondo' ,'Solaris' \
 ,'Verdelho' ,'Riesling' ,'Sangiovese']
conti = ['NAm', 'Eur', 'Sam' ,'Afr', 'Oce']

# selected countries/continents/varieties for Approach A

countries = ['USA' ,'Spain' ,'Italy' ,'Australia', 'Denmark']
varieties_kept = ['Cabernet Sauvignon','Chardonnay','Merlot','Sangiovese','Shiraz','Tempranillo']
conti = ['NAm' , 'Eur' , 'Oce']
# Select the response column to predict. Index 2 selects Grape_variety from this
# labels list. The chosen label and the filter lists must describe the same kind
# of value (for example, cultivar names must not be filtered against country names).

labels = ['Country' , 'Continent' , 'Grape_variety','comb','Rootstock']
class_label = labels[2]  # change index value accordingly
class_label

# Load the processed feature table. Rows are samples; feature columns are ASV
# measurements and metadata columns identify labels such as country or cultivar.
# The pickle file must be present in the process's current working directory.

file_name = './FeatureDataWoOut.pkl'
features_df = pd.read_pickle(file_name)
# X will contain predictor features and y will contain the selected response.
# The filters below all target features_df[class_label]. They are written as
# sequential filters, not alternatives: a row must pass every filter. Since the
# lists contain different kinds of labels, this can remove every row. For a given
# class_label, only the matching class-selection filter should normally be active.

## select one of the fllwong based on the class label used; this is sample filtering critera bason on taget labels

features_df =  features_df[features_df[class_label].isin(countries)]

features_df =  features_df[features_df[class_label].isin(varieties_kept)]

features_df =  features_df[features_df[class_label].isin(conti)]

# Additional exclusions for scion/rootstock analyses: remove unknown combinations
# and the own-rooted Shiraz combination, then exclude unknown and Shiraz rootstocks.
# These columns must exist in the input table; otherwise pandas raises a KeyError.
features_df =  features_df[~features_df['comb'].isin(['unknown','Shiraz-Shiraz'])]

features_df =  features_df[~features_df['Rootstock'].isin(['unknown','Shiraz'])]
# The code assumes the final six columns are metadata and every preceding column
# is a numeric ASV feature. If the table's column order changes, this slice may
# accidentally include metadata as predictors or omit real features.

feature_cols = list(features_df.columns[0:-6])


X = features_df[feature_cols]
y = features_df[class_label]

print(f' Shape of X = {str(X.shape)} and y = {str(y.shape)}')
print(y.unique())


# LABEL ENCODING MAPS
# Each dictionary maps readable class names to integer IDs. encode_maps is
# reassigned repeatedly below, so the last assignment that has executed is the
# one used by later calls to y.map(encode_maps). Keep the selected class label,
# map, and model target format paired with one another.

# for apprach A

countries_maps = {'USA':0 ,'Spain':1 ,'Croatia':2 ,\
                  'France':3 ,'Hungary':4 ,'Italy':5 \
                  ,'Argentina':6,'South Africa':7 \
                  ,'Australia':8, 'Denmark':9, 'Portugal':10 ,'Germany':11}
encode_maps = countries_maps

continent_maps = {'Afr': 0, 'Eur': 1, 'NAm': 2, 'Oce': 3, 'SAm': 4}
encode_maps = continent_maps

cultivar_maps = {'Airen': 0,
 'Barbera': 1,
 'Cabernet Franc': 2,
 'Cabernet Sauvignon': 3,
 'Callet': 4,
 'Chardonnay': 5,
 'Grenache': 6,
 'Malbec': 7,
 'Merlot': 8,
 'Pinot Noir': 9,
 'Riesling': 10,
 'Rondo': 11,
 'Sangiovese': 12,
 'Sauvignon Blanc': 13,
 'Shiraz': 14,
 'Solaris': 15,
 'Tempranillo': 16,
 'Verdejo': 17,
 'Verdelho': 18}

encode_maps = cultivar_maps

# Approach B maps. These assignments overwrite the corresponding Approach A
# mapping variables when execution reaches them.


countries_maps = {'Australia': 0, 'Denmark': 1, 'Italy': 2, 'Spain': 3, 'USA': 4}
encode_maps = countries_maps
continent_maps = {'Eur': 0, 'NAm': 1, 'Oce': 2}
encode_maps = continent_maps
cultivar_maps = {'Cabernet Sauvignon': 0,
 'Chardonnay': 1,
 'Merlot': 2,
 'Sangiovese': 3,
 'Shiraz': 4,
 'Tempranillo': 5}

encode_maps = cultivar_maps

# Mappings for scion/rootstock analyses. These are additional alternatives, not
# maps to apply consecutively. The final encode_maps assignment here is rstk_maps.

comb_maps = {'Chardonnay-110R': 0,
 'Chardonnay-3309C': 1,
 'Chardonnay-420A': 2,
 'Chardonnay-SO4': 3,
 'Merlot-3309C': 4,
 'Merlot-420A': 5,
 'Sangiovese-110R': 6,
 'Sangiovese-420A': 7}

encode_maps = comb_maps

culti_maps = {'Chardonnay': 0, 'Merlot': 1, 'Sangiovese': 2, 'Shiraz': 3}
encode_maps = culti_maps

rstk_maps = {'110R': 0, '3309C': 1, '420A': 2, 'SO4': 3}
encode_maps = rstk_maps

# CLASSICAL MACHINE-LEARNING MODELS
# The following blocks describe the intended classifiers: Random Forest, two
# Naive Bayes variants, Gradient Boosting, AdaBoost, linear and RBF SVMs, and
# k-nearest neighbors. Each block fits on X_train/y_train and predicts X_test.
# In this file those train/test variables are assigned later, so these cells
# require the notebook-style execution order or previously populated variables.
# Several estimator classes used below are also missing from the imports above.
''' ML models code the follwoeong code is for all ML models used in the analysis, after data filtering fofr appoach A or B is perfromed above code is same.
the code foR , Random Forests (RF), Adaptive Boost (ADA), Gradient Boost (GBM), Support Vector Machines with linear (SVML) and radial (SVMR) kernels, Gaussian (GNB) and Bernoulli Naïve Bayes (BNB), k-Nearest Neighbor (KNN) IS SAME 
plese note that the paarmeters for all the alo . were selected using repetd k-fold stratified CV. the NN apprah for A vs b are differnt are sepertly incded below
'''






############################

# RANDOM FOREST
# A forest averages many decision trees. n_estimators controls tree count;
# max_features limits candidate predictors at each split; max_depth limits tree
# growth; balanced_subsample adjusts class weights separately in each bootstrap
# sample. random_state makes the fit repeatable, and n_jobs=-1 uses available CPUs.
# The feature-selection object requests prefit=True, which normally requires an
# already-fitted estimator; rf_model is fitted only on the next line in this copy.

rf_model =  RandomForestClassifier(n_jobs=-1,random_state=5, \
                                 n_estimators = 1000, \
                                 max_features = 50000,
                                  max_depth = 250, \
                                 class_weight = "balanced_subsample") 
sel2 = SelectFromModel( rf_model, prefit = True )
#rf_model = RandomForestClassifier(n_jobs=-1,random_state=5)
rf_model.fit(X_train,y_train)
y_preds_rf = rf_model.predict(X_test)
#auc_ovr_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test), average='macro',multi_class='ovr')
#auc_ovo_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test),average = 'macro',multi_class='ovo')
f1_test_rf = f1_score(y_test,y_preds_rf,average='macro')

print(f'F1-macro Score: {f1_test_rf}')
#print(f'MCC Score: {matthews_corrcoef(y_test, y_preds_rf)}')
#print(f'AUC-macro OVR Score: {auc_ovr_rf}')
#print(f'AUC-macro OVO Score: {auc_ovo_rf}')
print(f'Blanced accuracy {balanced_accuracy_score(y_test,y_preds_rf)}')
print(classification_report(y_test, pd.Series(y_preds_rf), \
                            target_names=list(encode_maps.keys()), \
                            zero_division = 0  \
                            ))


# BERNOULLI NAIVE BAYES
# BernoulliNB models whether each feature is present/absent rather than its raw
# abundance. The transform below maps exactly zero to 0 and every nonzero value
# to 1. The prior list assumes exactly 12 classes and must match the actual class
# count and class ordering to be valid.

# defing a function for binarizing
def binarize_features(f):
    return np.where(f == 0, 0, 1)


X_train_bnb = X_train.apply(binarize_features)
X_test_bnb = X_test.apply(binarize_features)

bnb_priors = [1/12 for i in range(12)]
bnb_priors

bnb_base = BernoulliNB(force_alpha=True,binarize=None,class_prior = bnb_priors,fit_prior=False)
bnb_base.fit(X_train_bnb,y_train)
y_preds_bnb_base = bnb_base.predict(X_test_bnb)

f1_test_bnb_base = f1_score(y_test,y_preds_bnb_base,average='macro')
#auc_ovr_bnb_base = roc_auc_score(y_test, bnb_base.predict_proba(X_test_bnb), average='macro',multi_class='ovr')

print(f'F1-macro Score: {f1_test_bnb_base}')
print(f'MCC Score: {matthews_corrcoef(y_test, y_preds_bnb_base)}')
#print(f'AUC-macro Score: {auc_ovr_bnb_base}')
print(classification_report(y_test, pd.Series(y_preds_bnb_base), \
                            target_names=list(countries_maps.keys()), \
                            zero_division = 0  \
                            ))


# GaussianNB instead models continuous-valued features under a per-class Gaussian
# assumption, so it receives the original (not binarized) feature matrix.
gnb_base = GaussianNB()
gnb_base.fit(X_train,y_train)
y_preds_gnb_base = gnb_base.predict(X_test)

f1_test_gnb_base = f1_score(y_test,y_preds_gnb_base,average='macro')
auc_ovr_gnb_base = roc_auc_score(y_test, gnb_base.predict_proba(X_test), average='macro',multi_class='ovr')

print(f'F1-macro Score: {f1_test_gnb_base}')
print(f'AUC-macro Score: {auc_ovr_gnb_base}')
print(classification_report(y_test, pd.Series(y_preds_gnb_base), \
                            target_names=list(countries_maps.keys()), \
                            zero_division = 0  \
                            ))



# GRADIENT BOOSTING
# GradientBoostingClassifier adds trees sequentially, with each tree correcting
# errors made by the current ensemble. The fitted model is also passed to a
# SelectFromModel selector; predictions and macro-F1 are then computed on X_test.

gbm_base = GradientBoostingClassifier(random_state=5,max_depth = 250, max_features = 50000, n_estimators = 500)
gbm_base.fit(X_train,y_train)

sel9 = SelectFromModel( gbm_base, prefit = True )
y_preds_gbm_base = gbm_base.predict(X_test)

f1_test_gbm_base = f1_score(y_test,y_preds_gbm_base,average='macro')
#auc_ovr_gbm_base = roc_auc_score(y_test, gbm_base.predict_proba(X_test), average='macro',multi_class='ovr')

print(f'F1-macro Score: {f1_test_gbm_base}')
#print(f'MCC Score: {matthews_corrcoef(y_test, y_preds_gbm_base)}')
#print(f'AUC-macro Score: {auc_ovr_gbm_base}')
"""print(classification_report(y_test, pd.Series(y_preds_gbm_base), \
                            target_names=list(encode_maps.keys()), \
                            zero_division = 0  \
                            ))"""


# ADABOOST
# AdaBoost builds an ensemble by repeatedly emphasizing examples previous learners
# classified poorly. Here the base learner is a deep decision tree and 10,000
# boosting rounds can be computationally expensive. The commented-out reports
# are inactive and do not contribute to the results.

#ada_base = AdaBoostClassifier(random_state=5)
from sklearn.tree import DecisionTreeClassifier

ada_base = AdaBoostClassifier(n_estimators = 10000,estimator=DecisionTreeClassifier(max_depth=250,\
    max_features=45000,class_weight='balanced'),random_state=5)
ada_base.fit(X_train,y_train)

sel8 = SelectFromModel( ada_base, prefit = True )
y_preds_ada_base = ada_base.predict(X_test)

f1_test_ada_base = f1_score(y_test,y_preds_ada_base,average='macro')
#auc_ovr_ada_base = roc_auc_score(y_test, ada_base.predict_proba(X_test), average='macro',multi_class='ovr')

print(f'F1-macro Score: {f1_test_ada_base}')
#print(f'MCC Score: {matthews_corrcoef(y_test, y_preds_ada_base)}')
#print(f'AUC-macro Score: {auc_ovr_ada_base}')
"""print(classification_report(y_test, pd.Series(y_preds_ada_base), \
                            target_names=list(countries_maps.keys()), \
                            zero_division = 0  \
                            ))"""
                            
                            
# SUPPORT VECTOR MACHINES
# The first SVC uses a linear kernel and class_weight='balanced'; C=0.1 controls
# the penalty for margin violations. probability=True enables probability
# estimates but adds fitting overhead. The second SVC uses the default RBF kernel.

svml_base = SVC(random_state = 5,probability=True,kernel='linear',class_weight = 'balanced',gamma='scale',C=0.1)
svml_base.fit(X_train,y_train)
y_preds_svml_base = svml_base.predict(X_test)

f1_test_svml_base = f1_score(y_test,y_preds_svml_base,average='macro')
#auc_ovr_svml_base = roc_auc_score(y_test, svml_base.predict_proba(X_test), average='macro',multi_class='ovr')

print(f'F1-macro Score: {f1_test_svml_base}')
#print(f'MCC Score: {matthews_corrcoef(y_test, y_preds_svml_base)}')
#print(f'AUC-macro Score: {auc_ovr_svml_base}')
print(classification_report(y_test, pd.Series(y_preds_svml_base), \
                            target_names=list(encode_maps.keys()), \
                            zero_division = 0  \
                            ))


svm_base = SVC(random_state = 5,probability=True,kernel='rbf')
svm_base.fit(X_train,y_train)
y_preds_svm_base = svm_base.predict(X_test)

f1_test_svm_base = f1_score(y_test,y_preds_svm_base,average='macro')
#auc_ovr_svm_base = roc_auc_score(y_test, svm_base.predict_proba(X_test), average='macro',multi_class='ovr')

print(f'F1-macro Score: {f1_test_svm_base}')
#print(f'MCC Score: {matthews_corrcoef(y_test, y_preds_svm_base)}')
#print(f'AUC-macro Score: {auc_ovr_svm_base}')
print(classification_report(y_test, pd.Series(y_preds_svm_base), \
                            target_names=list(encode_maps.keys()), \
                            zero_division = 0  \
                            ))


# K-NEAREST NEIGHBORS
# This classifier assigns a sample the majority label among its three nearest
# training samples. It uses the original feature scale, so scaling can strongly
# affect distances; this script does not scale the features in this block.

knn_base = KNeighborsClassifier(n_neighbors=3,n_jobs=-1)
knn_base.fit(X_train,y_train)
y_preds_knn_base = knn_base.predict(X_test)

f1_test_knn_base = f1_score(y_test,y_preds_knn_base,average='macro')

print(f'F1-macro Score: {f1_test_knn_base}')
print(classification_report(y_test, pd.Series(y_preds_knn_base), \
                            target_names=list(cultivar_maps.keys()), \
                            zero_division = 0  \
                            ))



# NEURAL NETWORK: APPROACH A / SINGLE HOLDOUT
# The code below seeds TensorFlow, records feature/class counts, and creates a
# reproducible 80/20 split. A second split is performed after one-hot encoding.
# Note that the classical-model blocks above occur before this split in the file.

tf.random.set_seed(5)
n_features = X.shape[1]
n_classes = len(np.unique(y))
print(n_features,n_classes)


## train/test Split
print(f' Shape of X = {str(X.shape)} and y = {str(y.shape)}')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8,stratify=y)
print(f' Shape of X_train = {str(X_train.shape)} and y_train = {str(y_train.shape)}')
print(f' Shape of X_test = {str(X_test.shape)} and y_test = {str(y_test.shape)}')
print(np.unique(y_train))
print(np.unique(y_test))




# Convert readable class labels to the integer IDs defined by encode_maps. This
# requires every observed value in y to be a key in the currently active map;
# unmatched values become missing values (NaN) and can break later training.
y = y.map(encode_maps)
y_original = y.copy()
print(np.unique(y))

# One-hot encoding converts each integer class ID into a vector with one active
# position. This representation is required by the later categorical cross-
# entropy model, but it is not the integer-label format expected by sparse
# categorical cross-entropy or many scikit-learn classification metrics.
y = tf.convert_to_tensor(y)
one_hot_y = tf.one_hot(y, n_classes)
one_hot_y = one_hot_y.numpy()
one_hot_y = pd.DataFrame(one_hot_y)
print(one_hot_y)

# Make the 80/20 split using the one-hot target matrix for stratification. This
# overwrites X_train/X_test/y_train/y_test from the earlier integer-label split.
# For classification evaluation, keep the corresponding integer class IDs too.
print(f' Shape of X = {str(X.shape)} and y = {str(one_hot_y.shape)}')
X_train, X_test, y_train, y_test = train_test_split(X, one_hot_y, test_size=0.2, random_state=8,stratify=one_hot_y)
print(f' Shape of X_train = {str(X_train.shape)} and y_train = {str(y_train.shape)}')
print(f' Shape of X_test = {str(X_test.shape)} and y_test = {str(y_test.shape)}')
print(np.unique(y_train,axis = 0))
print(np.unique(y_test, axis = 0))

# Recover each test sample's class index from its one-hot row. This is the label
# form needed to compare with argmax predictions later, although y_test itself
# remains one-hot encoded in the following fit/evaluation calls.

y_test_index = tf.argmax(y_test, axis=1)
y_test_index = pd.Series(y_test_index.numpy())
print(y_test_index)


# APPROACH A MODEL ARCHITECTURES
# Both functions below have the same Python name. The second definition replaces
# the first one, so model = build_model() uses only the second architecture.
# The first architecture is a smaller, repeated 800-unit design intended for
# continent labels; the second is a wider/deeper design intended for cultivar or
# country labels. Dense layers learn nonlinear combinations of ASV features,
# Dropout randomly disables units during training, and L2 regularization penalizes
# large weights. The final softmax layer returns one score per class.

# First architecture (continent version). SparseCategoricalCrossentropy expects
# integer class IDs, whereas the preceding split sets y_train to one-hot rows.
# This definition is also replaced by the second build_model definition below.
def build_model(nclasses = n_classes):
  # define model
  model = Sequential()
  model.add(Dense(800, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
  model.add(Dense(800, activation='relu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(800, activation='relu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.001)))
  model.add(Dense(800, activation='relu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(800, activation='relu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.001)))
  model.add(Dense(800, activation='relu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.001)))
  model.add(Dense(nclasses, activation='softmax'))
  # compile the model
  opt = SGD(learning_rate=0.001)
  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=opt, metrics=['accuracy'])
  # fit the model
  return model


# Second architecture (cultivar/country version). This is the active definition
# after this line. opt2 is instantiated but not passed to compile; SGD is the
# optimizer actually used. Its sparse loss also expects integer labels, which do
# not match the one-hot y_train created above unless labels are converted first.

def build_model(nclasses = n_classes):
  # define model
  model = Sequential()
  model.add(Dense(1000, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
  model.add(Dense(1200, activation='relu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.001)))
  model.add(Dense(1500, activation='relu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.001)))
  model.add(Dropout(0.3))
  model.add(Dense(1000, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(0.001)))
  model.add(Dense(1200, activation='relu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.001)))
  model.add(Dense(1500, activation='relu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.001)))
  model.add(Dropout(0.3))
  model.add(Dense(1000, activation='relu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.001)))
  model.add(Dense(800, activation='relu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(1200, activation='relu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.001)))
  model.add(Dense(800, activation='relu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.001)))
  model.add(Dense(nclasses, activation='softmax'))
  # compile the model
  opt = SGD(learning_rate=0.001)
  opt2 = tf.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=opt, metrics=['accuracy'])
  # fit the model
  return model

# Instantiate the active (second) architecture and print its layer summary.
model = build_model()
print(model.summary())

BATCH_SIZE = 8
EPOCHS = 10

# Compute class weights so errors on less frequent classes can receive more
# emphasis. This API expects a one-dimensional vector of class IDs; y_train here
# is a one-hot DataFrame, so the target representation should be checked before
# relying on the resulting weights. The dictionary is passed to Keras fit below.
cw = sklearn.utils.class_weight.compute_class_weight('balanced' ,classes=np.unique(y_train) ,y=y_train)
cw = dict(enumerate(cw.flatten(), 0))
cw


# Train for at most EPOCHS passes through the training data. No validation_data
# or validation_split is supplied here, so history normally has no val_accuracy
# series even though the next plotting block tries to read one.
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,class_weight=cw)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# HOLDOUT EVALUATION
# Evaluate loss/accuracy on the test and training partitions, then predict test
# classes by taking the largest output score. For consistent classification
# metrics, y_test should be represented as class IDs to match y_preds; as written,
# y_test was left one-hot encoded by the previous split.

loss, acc = model.evaluate(X_test, y_test, verbose=1)

loss2, acc2 = model.evaluate(X_train, y_train, verbose=1)

raw_preds = model.predict(X_test, batch_size=BATCH_SIZE,verbose=0)
#print(raw_preds)
y_preds = [np.argmax(i) for i in raw_preds]
score_f1 = f1_score(y_test,y_preds,average='macro')
print(f'f1 macro: {score_f1}')
print(f'bal acc: {balanced_accuracy_score(y_test,y_preds)}')

print(classification_report(y_test, pd.Series(y_preds), \
                            target_names=list(encode_maps.keys()), \
                            zero_division = 0  \
                            ))




# CONFUSION MATRICES
# Convert class IDs back to their readable names, then plot a row-normalized
# confusion matrix (each true-class row sums to one) and a raw-count matrix.
# The mapping must correspond to the encoding used to train this specific model.

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

encode_maps_inv = {v: k for k,v in encode_maps.items()}
encode_maps_inv

y_preds = pd.Series(y_preds)
mapss_inv = encode_maps_inv
cm = confusion_matrix(y_test.map(mapss_inv),y_preds.map(mapss_inv),labels=np.unique(y_train.map(mapss_inv)),normalize = 'true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.unique(y_train.map(mapss_inv)))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)
#disp.plot()
#plt.tight_layout()


#plt.savefig('RF_base_country_best2.png')
plt.show()


cm2 = confusion_matrix(y_test.map(mapss_inv),y_preds.map(mapss_inv),labels=np.unique(y_train.map(mapss_inv)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=np.unique(y_train.map(mapss_inv)))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)
#disp.plot()
plt.tight_layout()


#plt.savefig('RF_base_country_best2.png')
plt.show()


# APPROACH B: NEURAL-NETWORK ARCHITECTURE SEARCH
# This second workflow uses the stricter class set and compares networks with
# 1-3 hidden layers and 8-1024 units per layer. It is a separate analysis from
# the single holdout workflow above and resets several shared variables.

# Re-imports used by the notebook cell from which this block was taken. Repeated
# imports do not change behavior, but this cell depends on TensorFlow, pandas,
# NumPy, garbage collection, and scikit-learn being available in the environment.
from numpy import mean
from numpy import std
import numpy as np
import pandas as pd
import gc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import RepeatedStratifiedKFold


# Select the response for Approach B. Here index 2 means the column named
# Cultivar; that column must exist in the input feature table.
labels = ['Country' , 'Continent' , 'Cultivar']
class_label = labels[2]
file_name = './FeatureDataWoOut.pkl'
features_df = pd.read_pickle(file_name)

feature_cols = list(features_df.columns[0:-6])

# The active slice again assumes the last six columns are metadata. The commented
# alternative would treat all but the final column as features, a different schema.

X = features_df[feature_cols]
y = features_df[class_label]

tf.random.set_seed(5)
n_features = X.shape[1]
n_classes = len(np.unique(y))
print(n_features,n_classes)

print(f' Shape of X = {str(X.shape)} and y = {str(y.shape)}')
print(y.unique())

cultivar_maps = {'Cabernet Sauvignon': 0,
 'Chardonnay': 1,
 'Merlot': 2,
 'Sangiovese': 3,
 'Shiraz': 4,
 'Tempranillo': 5}

encode_maps = cultivar_maps

# Map the six retained cultivar names to stable integer IDs. The mapping and
# class_label must agree; otherwise y.map produces missing targets.
y = y.map(encode_maps)
y_original = y.copy()
print(np.unique(y))

# Convert integer IDs to one-hot vectors because this Approach B model uses
# CategoricalFocalCrossentropy, which is configured below for categorical targets.
y = tf.convert_to_tensor(y)
one_hot_y = tf.one_hot(y, n_classes)
one_hot_y = one_hot_y.numpy()
one_hot_y = pd.DataFrame(one_hot_y)
print(one_hot_y)

# Hold out 20 percent of samples. Stratification uses one-hot class rows so the
# class proportions are approximately retained in each partition.
print(f' Shape of X = {str(X.shape)} and y = {str(one_hot_y.shape)}')
X_train, X_test, y_train, y_test = train_test_split(X, one_hot_y, test_size=0.2, random_state=8,stratify=one_hot_y)
print(f' Shape of X_train = {str(X_train.shape)} and y_train = {str(y_train.shape)}')
print(f' Shape of X_test = {str(X_test.shape)} and y_test = {str(y_test.shape)}')
print(np.unique(y_train,axis = 0))
print(np.unique(y_test, axis = 0))

# decode the y_test for evaluation
# decode the ytest 

y_test_index = tf.argmax(y_test, axis=1)
y_test_index = pd.Series(y_test_index.numpy())
print(y_test_index)

# Disable TensorFlow eager execution globally for this process. This is a
# process-wide setting and can affect later TensorFlow code, including SHAP; it
# should only be enabled when required by the installed TensorFlow/API version.
tf.compat.v1.disable_eager_execution()

# Stop a fit when validation loss has not improved for three epochs. The callback
# below receives each cross-validation test fold as validation_data; consequently
# that fold both chooses the stopping epoch and supplies the reported score.
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)


EPOCHS_MAX = 20
BATCH_SIZE = 8
# Search all 24 combinations formed by three hidden-layer counts and eight layer
# widths. Dropout is enabled after every hidden layer when use_dropout is True.
use_dropout = True
layers_list = list()
nodes_list = list()
for l in [1,2,3]:
    for n in [8,16,32,64,128,256,512,1024]:
        layers_list.append(l)
        nodes_list.append(n)
        
final_results_f1_macro = list()
n_epochs_global = list()

for iter in range(len(layers_list)):
    # One iteration trains one architecture configuration across all CV folds.
    print(f'iteration : {iter}')
    outer_results_f1_macro = list()
    n_epochs = list()
    # Five stratified folds are used once. random_state makes fold assignment
    # repeatable; n_repeats=1 means this is ordinary 5-fold stratified CV.
    cv_nn = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)

    # xx indexes training samples and yy indexes the held-out fold. The same yy
    # fold is used for early stopping and F1 scoring, so these scores are not a
    # fully independent estimate of generalization performance.
    for xx, yy in cv_nn.split(X,y_original):
        tf.keras.backend.clear_session()
        X_train2, y_train2 = X.iloc[xx],one_hot_y.iloc[xx]
        X_test2,y_test2,y_test_index2 = X.iloc[yy],one_hot_y.iloc[yy],y_original.iloc[yy]
        print('building model')
        my_mod = Sequential()
        # Add the requested number of hidden layers. The first layer declares the
        # input width; later layers repeat the selected width and use L2 penalty.
        for layer in range(layers_list[iter]):
                if layer == 0:
                    my_mod.add(Dense(nodes_list[iter], activation='relu', input_shape=(n_features,)))
                else:
                    my_mod.add(Dense(nodes_list[iter], activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.001)))
                if use_dropout:
                    my_mod.add(Dropout(0.2))
        my_mod.add(Dense(n_classes, activation='softmax'))
        # Focal loss down-weights easy examples and emphasizes hard examples.
        # gamma controls that emphasis and alpha adjusts class weighting; Adam
        # performs optimization. Targets are one-hot matrices in this block.
        my_mod.compile(loss=tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, alpha=0.5),optimizer='adam', metrics=['accuracy'])
        print('fitting model ')
        his = my_mod.fit(X_train2, y_train2, validation_data= (X_test2,y_test2), callbacks=[callback], epochs=EPOCHS_MAX, batch_size=BATCH_SIZE, verbose=0)
        # Record how many epochs ran before early stopping (or the epoch limit).
        n_epochs.append(len(his.history['loss']))
        print('predicting')
        raw_preds2 = my_mod.predict(X_test2, batch_size=BATCH_SIZE,verbose=0)
        #print(raw_preds)
        y_preds2 = [np.argmax(i) for i in raw_preds2]
        score_f1 = f1_score(y_test_index2,y_preds2,average='macro')
        #print(f'f1 macro: {score_f1}')
        outer_results_f1_macro.append(score_f1)
        del my_mod
        gc.collect()
    
    
    # Despite the printed word "Accuracy", these values are the mean and
    # standard deviation of macro-F1 across the five folds.
    print('Accuracy: %.3f (%.3f)' % (mean(outer_results_f1_macro), std(outer_results_f1_macro)))
    final_results_f1_macro.append(mean(outer_results_f1_macro))
    n_epochs_global.append(n_epochs)

# Collect one row per architecture, including its mean macro-F1 and the list of
# epochs used in each fold, then sort the table from highest to lowest score.
pd.DataFrame({'layers': layers_list, 'nodes': nodes_list, 'score' : final_results_f1_macro,'epochs':n_epochs_global}).sort_values(by='score', ascending=False)




# SHAP FEATURE ATTRIBUTION
# DeepExplainer estimates each input feature's contribution to model outputs.
# The code assumes `model`, `X_train`, and `X_test` still refer to the selected
# trained model and its matching cultivar feature matrices after the CV block.


import shap
import numpy as np


preds = model.predict(X_test)

# Use up to the first 100 training samples as the SHAP background reference; this
# reference defines the baseline output against which feature contributions are
# estimated. A representative background sample is important for interpretation.
explainer = shap.DeepExplainer(model, X_train[:100])

X_test.head()

# Explain every test sample. Additivity checking is disabled, so SHAP will not
# raise its usual consistency warning if contributions do not sum to model output.
shap_values_all = explainer.shap_values(X_test.values, check_additivity=False)

n_classes = model.output_shape[1] 

top_asvs_per_cultivar = {}

for class_index in range(n_classes):
    if isinstance(shap_values_all, np.ndarray):
        # (n_samples, n_features, n_classes)
        shap_vals_class = shap_values_all[:, :, class_index]
    else:
        # List of arrays (n_samples, n_features)
        shap_vals_class = shap_values_all[class_index]

    # Retain positive contributions only: these push the model output toward this
    # class under SHAP's sign convention. Negative evidence is deliberately omitted.
    mean_positive_shap = np.mean(np.where(shap_vals_class > 0, shap_vals_class, 0), axis=0)

    # Select the 20 features with the largest mean positive SHAP value for this
    # class. The print label below says "Top 10" but the slice actually returns 20.
    top_idx = np.argsort(mean_positive_shap)[::-1][:20]
    top_asvs_per_cultivar[class_index] = np.array(X_test.columns)[top_idx]

    print(f"Top 10 ASVs for cultivar {class_index}:")
    print(top_asvs_per_cultivar[class_index])
    print("-"*50)

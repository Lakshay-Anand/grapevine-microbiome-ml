#### Importing Libraries 


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

tf.config.list_physical_devices('GPU')

'''
We used two different filtering criteria for sparse classes to build the ML models.
Approach A: Inclusion of Sparse Classes: In this approach, any class with a minimum of 3 samples was retained, maximizing the number of classes at the cost of potential over-representation of the majority class(es). 
Approach B: Exclusion of Sparse Classes: Here, only classes with at least 15 samples were retained. This criterion ensured a more balanced representation of classes for robust evaluation at the expense of reduced class diversity. 
'''

#### defening labels and variables 

''' for preventing code erdudancy, variables are set according to the two different labels "class_label" . selected countries, continents, and varieties are defined for both Approach A and Approach B below '''

# selected countries/continents/varieties for Approach A


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
# change according to the class label used 

labels = ['Country' , 'Continent' , 'Grape_variety','comb','Rootstock']
class_label = labels[2]  # change index value accordingly
class_label

# input data stored as pickle this is processed micobiome matrix with features (ASVs) as columns and samples as rows

file_name = './FeatureDataWoOut.pkl'
features_df = pd.read_pickle(file_name)
### constructing X and y from data

## select one of the fllwong based on the class label used; this is sample filtering critera bason on taget labels

features_df =  features_df[features_df[class_label].isin(countries)]

features_df =  features_df[features_df[class_label].isin(varieties_kept)]

features_df =  features_df[features_df[class_label].isin(conti)]

# for scion-rootstock combinatio a seprate metadat column 'comb' has the infomation for the scion-rootstoc and rootsck follwing filtering criteria is used which remove any unknown roosttstock sample and own-rooted Shiraz sample
features_df =  features_df[~features_df['comb'].isin(['unknown','Shiraz-Shiraz'])]

features_df =  features_df[~features_df['Rootstock'].isin(['unknown','Shiraz'])]
# extract features columns only

feature_cols = list(features_df.columns[0:-6])


X = features_df[feature_cols]
y = features_df[class_label]

print(f' Shape of X = {str(X.shape)} and y = {str(y.shape)}')
print(y.unique())


## coverting labels into numbers/indexes

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

# for appach B


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

# fo r scion, scion-rootstock and rootstock analysis

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

''' ML models code the follwoeong code is for all ML models used in the analysis, after data filtering fofr appoach A or B is perfromed above code is same.
the code foR , Random Forests (RF), Adaptive Boost (ADA), Gradient Boost (GBM), Support Vector Machines with linear (SVML) and radial (SVMR) kernels, Gaussian (GNB) and Bernoulli Naïve Bayes (BNB), k-Nearest Neighbor (KNN) IS SAME 
plese note that the paarmeters for all the alo . were selected using repetd k-fold stratified CV. the NN apprah for A vs b are differnt are sepertly incded below
'''






############################

## random forest

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


## Naive Bayes

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



### Gradient Boost 

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


## adaBoost

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
                            
                            
# SVM 

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


# kNN

knn_base = KNeighborsClassifier(n_neighbors=3,n_jobs=-1)
knn_base.fit(X_train,y_train)
y_preds_knn_base = knn_base.predict(X_test)

f1_test_knn_base = f1_score(y_test,y_preds_knn_base,average='macro')

print(f'F1-macro Score: {f1_test_knn_base}')
print(classification_report(y_test, pd.Series(y_preds_knn_base), \
                            target_names=list(cultivar_maps.keys()), \
                            zero_division = 0  \
                            ))



### Neural Network apprach A

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




# enconding the classes as integers 
y = y.map(encode_maps)
y_original = y.copy()
print(np.unique(y))

# one-hot encode y 
y = tf.convert_to_tensor(y)
one_hot_y = tf.one_hot(y, n_classes)
one_hot_y = one_hot_y.numpy()
one_hot_y = pd.DataFrame(one_hot_y)
print(one_hot_y)

# splitting for one-hot encoded 
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


## NN Model Architecture 

# for contienent
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


# for cultivar and country

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

model = build_model()
print(model.summary())

BATCH_SIZE = 8
EPOCHS = 10

## class weight adjust for class imbalance 
cw = sklearn.utils.class_weight.compute_class_weight('balanced' ,classes=np.unique(y_train) ,y=y_train)
cw = dict(enumerate(cw.flatten(), 0))
cw


history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,class_weight=cw)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


### Testing model 

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




## Confusion matrices 

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


# Approach B 

# approach B neural network 
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


labels = ['Country' , 'Continent' , 'Cultivar']
class_label = labels[2]
file_name = './FeatureDataWoOut.pkl'
features_df = pd.read_pickle(file_name)

feature_cols = list(features_df.columns[0:-6])

#feature_cols = list(features_df.columns[0:-1])

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

# enconding the classes as integers 
y = y.map(encode_maps)
y_original = y.copy()
print(np.unique(y))

# one-hot encode y 
y = tf.convert_to_tensor(y)
one_hot_y = tf.one_hot(y, n_classes)
one_hot_y = one_hot_y.numpy()
one_hot_y = pd.DataFrame(one_hot_y)
print(one_hot_y)

# splitting for one-hot encoded 
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

# disable eager executation (only for focal )
tf.compat.v1.disable_eager_execution()

# define A Callback for early stopping if validation loss does not change 
# for three consecutive epochs
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)


EPOCHS_MAX = 20
BATCH_SIZE = 8
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
    # code for each run
    print(f'iteration : {iter}')
    outer_results_f1_macro = list()
    n_epochs = list()
    cv_nn = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)

    for xx, yy in cv_nn.split(X,y_original):
        tf.keras.backend.clear_session()
        X_train2, y_train2 = X.iloc[xx],one_hot_y.iloc[xx]
        X_test2,y_test2,y_test_index2 = X.iloc[yy],one_hot_y.iloc[yy],y_original.iloc[yy]
        print('building model')
        my_mod = Sequential()
        for layer in range(layers_list[iter]):
                if layer == 0:
                    my_mod.add(Dense(nodes_list[iter], activation='relu', input_shape=(n_features,)))
                else:
                    my_mod.add(Dense(nodes_list[iter], activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.001)))
                if use_dropout:
                    my_mod.add(Dropout(0.2))
        my_mod.add(Dense(n_classes, activation='softmax'))
        my_mod.compile(loss=tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, alpha=0.5),optimizer='adam', metrics=['accuracy'])
        print('fitting model ')
        his = my_mod.fit(X_train2, y_train2, validation_data= (X_test2,y_test2), callbacks=[callback], epochs=EPOCHS_MAX, batch_size=BATCH_SIZE, verbose=0)
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
    
    
    print('Accuracy: %.3f (%.3f)' % (mean(outer_results_f1_macro), std(outer_results_f1_macro)))
    final_results_f1_macro.append(mean(outer_results_f1_macro))
    n_epochs_global.append(n_epochs)

# results 
pd.DataFrame({'layers': layers_list, 'nodes': nodes_list, 'score' : final_results_f1_macro,'epochs':n_epochs_global}).sort_values(by='score', ascending=False)




## SHAP analysis from the best performing model


import shap
import numpy as np


preds = model.predict(X_test)

explainer = shap.DeepExplainer(model, X_train[:100])

X_test.head()

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

    # Keep only positive contributions
    mean_positive_shap = np.mean(np.where(shap_vals_class > 0, shap_vals_class, 0), axis=0)

    # Top 10 ASVs
    top_idx = np.argsort(mean_positive_shap)[::-1][:20]
    top_asvs_per_cultivar[class_index] = np.array(X_test.columns)[top_idx]

    print(f"Top 10 ASVs for cultivar {class_index}:")
    print(top_asvs_per_cultivar[class_index])
    print("-"*50)

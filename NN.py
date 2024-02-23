######################################################################
####### hyper-parameter optimisation for NN ##########################
######### @author : lakshay.anand@uky.edu ############################
######################################################################


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
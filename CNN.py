import tensorflow as tf
import numpy as np
import pandas as pd
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding1D, BatchNormalization, Flatten, Conv1D
from keras.layers import AveragePooling1D, MaxPooling1D, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras import optimizers
from keras.models import Sequential
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,f1_score,recall_score
from sklearn.metrics import mean_squared_error
data = pd.read_csv('E:\\MaxC_0_1e_5_lnr-20180303T013254Z-001\\MaxC_0_1e_5_lnr\\spec_C2H6_all_9_pxl_1000_samples_1000_SNR_300_dB.csv')

y_vars = ['C2H6','CH4','CO','H2O','HBr','HCl','HF','N2O','NO']
y = data[y_vars]
data_vars = data.columns.values.tolist()
X_vars=[i for i in data_vars if i not in y_vars]
X = data[X_vars]

X = np.expand_dims(X, axis=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

def TrainModel(input_shape):
    X_input = Input(input_shape)
    #X = Dropout(rate=0.2, seed=2)(X_input)
    X = ZeroPadding1D(3)(X_input)
    X = Conv1D(32,  5, strides=1, name='conv0')(X)
    X = BatchNormalization(axis=2, name='bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(3, name='max_pool0')(X)


    X = ZeroPadding1D(3)(X)
    X = Conv1D(32, 2, strides=1, name='conv1')(X)
    X = BatchNormalization(axis=2, name='bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(2, name='max_pool1')(X)

    X = Flatten()(X)
    X = Dense(1024, activation='selu', name='fc1')(X)
    X = Flatten()(X)
    X = Dense(9, activation='softmax', name='fc2')(X)

    model = Model(inputs=X_input, outputs=X, name='TrainModel')
    return model

def generate_t(X, Y):
    m = X.shape[0];
    t = np.zeros(shape=m)
    Yt = Y.copy()
    Yt[Y>0] = 1
    Yt = np.array(Yt)
    for i in range(m):
        f1 = np.zeros(9)
        x = X[i]
        y = Yt[i]   #one_hot y_train
        x = np.multiply(x, y)
        for j in range(9):
            if x[j]>0:
                candidate = x[j]
                x1 = x.copy()
                x1[x1>= candidate] = 1
                x1[x1 < candidate] = 0
                f1[j] = f1_score(y_true=y, y_pred=x1)
        index = np.argmax(f1)
        t[i] = x[index]
    t = np.transpose([t])
    return t

def evaluate(y_test, y_pre):
    y_t = y_test.copy()
    y_p = y_pre.copy()
    y_t[y_t > 0] = 1
    y_p[y_p > 0] = 1
    precision_micro = precision_score(y_true=y_t, y_pred=y_p, average='micro')
    precision_macro = precision_score(y_true=y_t, y_pred=y_p, average='macro')
    recall_micro = recall_score(y_true=y_t, y_pred=y_p, average='micro')
    recall_macro = recall_score(y_true=y_t, y_pred=y_p, average='macro')
    f1_micro = f1_score(y_true=y_t, y_pred=y_p, average='micro')
    f1_macro = f1_score(y_true=y_t, y_pred=y_p, average='macro')
    return precision_micro,precision_macro,recall_micro,recall_macro,f1_micro,f1_macro

trainModel = TrainModel(X_train.shape[1:])
trainModel.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
trainModel.fit(X_train, y_train, epochs=50, batch_size=10)
probability = trainModel.predict(X_test, verbose=1)
#probability_train = trainModel.predict(X_train, verbose=1)
#t_train = generate_t(probability_train, y_train)
preds = trainModel.evaluate(X_test, y_test, batch_size=32, verbose=1, sample_weight=None)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


#Quantification
a = X_train.shape[1:]
a = a[0]
X_train = np.reshape(X_train,(-1,a))
X_test = np.reshape(X_test,(-1,a))
Quantification_model = linear_model.LinearRegression()
Quantification_model.fit(X_train, y_train)
Quanti_predict = Quantification_model.predict(X_test)


#label-prediction
'''Threshold_model = linear_model.LinearRegression()
Threshold_model.fit(probability_train, t_train)
Threshold_predict = Threshold_model.predict(probability)
Threshold = np.multiply(np.ones(probability.shape),Threshold_predict)'''
Threshold = 0.03*np.ones(probability.shape)
L_predict = np.greater(probability,Threshold)

prediction = np.multiply(L_predict, Quanti_predict)
prediction_score = mean_squared_error(y_test, prediction)
print('Mean square error for quantification:',prediction_score)


pi,pa,ri,ra,fi,fa = evaluate(y_test,L_predict)
print('precision_micro:',pi)
print('precision_macro:',pa)
print('recall_micro:',ri)
print('recall_macro:',ra)
print('f1_score_micro:',fi)
print('f1_score_macro:',fa)















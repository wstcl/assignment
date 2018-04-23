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

data = pd.read_csv('E:\\MaxC_0_1e_5_lnr-20180303T013254Z-001\\MaxC_0_1e_5_lnr\\spec_C2H6_all_9_pxl_1000_samples_1000_SNR_1_dB.csv')

y_vars = ['C2H6','CH4','CO','H2O','HBr','HCl','HF','N2O','NO']
y = data[y_vars]
data_vars = data.columns.values.tolist()
X_vars=[i for i in data_vars if i not in y_vars]
X = data[X_vars]

X = np.expand_dims(X, axis=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

def TrainModel(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding1D(3)(X_input)
    X = Conv1D(32,  7, strides=1, name='conv0')(X)
    X = BatchNormalization(axis=2, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling1D(2, name='max_pool')(X)
    X = Flatten()(X)
    X = Dense(9, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='TrainModel')
    return model

trainModel = TrainModel(X_train.shape[1:])
trainModel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
trainModel.fit(X_train, y_train, epochs=40, batch_size=50)
b = trainModel.predict_classes(X_test, verbose=1)
preds = trainModel.evaluate(X_test, y_test, batch_size=32, verbose=1, sample_weight=None)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
a = X_train.shape[1:]
a = a[0]
X_train = np.reshape(X_train,(-1,a))
X_test = np.reshape(X_test,(-1,a))
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
predict = model.predict(X_test)




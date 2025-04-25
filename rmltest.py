import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import pandas as pd
import tensorflow as tf
#from keras.utils import np_utils
import tensorflow.keras.models as models
from keras.layers import Reshape, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, GaussianNoise, BatchNormalization
from keras.regularizers import *
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, random, sys, keras
from keras_tqdm import TQDMNotebookCallback
from sklearn.model_selection import train_test_split
from sklearn import metrics
import h5py
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)
print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

filename = "/home/017448899/RML2018/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"

def load_hdf5(filename):
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        keys = list(f.keys())
        X = f.get(keys[0])[:]
        Y = f.get(keys[1])[:]
        Z = f.get(keys[2])[:]
    return X, Y, Z

X, Y, Z = load_hdf5(filename) # this may take up to a minute
print(X.shape) # Input (radio signal data)
print(Y.shape) # Target values (in one-hot encoding)
print(Z.shape) # Signal to noise ratios
print('')
print("Signal to noise ratio")
print(np.unique(Z, return_counts=True)[0])
print("{} ratios with {} samples each".format(np.unique(Z).shape[0], np.unique(Z, return_counts=True)[1][0]))
print('')
classes = np.array(['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM', '128APSK',
 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM'])
easy_classes = np.array([22, 11, 8, 19, 9, 23, 10, 16, 3, 4, 6])
print("Easy Modulations: {}".format(classes[easy_classes]))
print("All Modulations: {}".format(classes))

indices = np.where(Z>=-14)[0]
X = X[indices]
Y = Y[indices]
Z = Z[indices]

print("X.shape: {}".format(X.shape))
print("Y.shape: {}".format(Y.shape))

# Partition into training and test datasets (https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros)
def get_train_test_inds(y,train_proportion=0.5):
    '''Generates indices, making random stratified split into training set and testing sets
    with proportions train_proportion and (1-train_proportion) of initial sample.
    y is any iterable indicating classes of each observation in the sample.
    Initial proportions of classes inside training and 
    testing sets are preserved (stratified sampling).
    '''

    y=np.array(y)
    train_inds = np.zeros(len(y),dtype=bool)
    test_inds = np.zeros(len(y),dtype=bool)
    values = np.unique(y, axis=0)
    for value in values:
        value_inds = (y==value).all(axis=1).nonzero()[0]
        np.random.shuffle(value_inds)
        n = int(train_proportion*len(value_inds))

        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True

    return train_inds,test_inds

train_inds,test_inds = get_train_test_inds(Y,train_proportion=0.5)
X_train = X[train_inds]
Y_train = Y[train_inds]
X_test = X[test_inds]
Y_test = Y[test_inds]
in_shp = list(X_train.shape[1:])

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
Y_train = Y_train.astype(np.float32)
Y_test = Y_test.astype(np.float32)

batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), Y_train.astype(np.float32)))
train_dataset = train_dataset.shuffle(buffer_size=1024)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_test.astype(np.float32), Y_test.astype(np.float32)))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)



print("X_train.shape: {}".format(X_train.shape))
print("Y_train.shape: {}".format(Y_train.shape))
print("X_test.shape: {}".format(X_test.shape))
print("Y_test.shape: {}".format(Y_test.shape))
print("input shape: {}".format(in_shp))

keras.backend.clear_session()

dr = 0.5 # dropout rate 
lr = 0.001 # learing rate
l2_reg = 0.00

model = models.Sequential()
model.add(Reshape([1]+in_shp, input_shape=in_shp))
# Layer 1
model.add(Conv2D(64, (1, 3), padding="same", name="conv1", kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
# Layer 2
model.add(Conv2D(64, (1, 3), padding="same", name="conv2", kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
# Layer 3
model.add(Conv2D(64, (1, 3), padding="same", name="conv3", kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
# Layer 4
model.add(Conv2D(64, (1, 3), padding="same", name="conv4", kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
# Layer 5
model.add(Conv2D(64, (1, 3), padding="same", name="conv5", kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
# Layer 6
model.add(Conv2D(64, (1, 3), padding="same", name="conv6", kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(1, 16)))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense(len(classes), kernel_initializer='he_normal', name="dense2"))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))

opt = keras.optimizers.Adam(learning_rate=lr)
model.compile(loss='categorical_crossentropy', optimizer=opt,  metrics=['accuracy'])
model.summary()


# Set up some params 
nb_epoch = 100     # number of epochs to train on
batch_size = 32  # training batch size, use highest possible power of two without getting oom


# perform training ...
#   - call the main training loop in keras for our network+dataset
filepath = 'rml2018.wts.h5'
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=nb_epoch,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    ],
    verbose=1
)
# we re-load the best weights once training is finished
model.load_weights(filepath)

# Takes 20-30 mins for one SNR: 10
# 74 mins for 4 SNRs: 12, 18, 24, 30
# 6-7 hours for 16 SNRs: -12 - 18
# 5 hours for 22 SNRs: -12 - 30
# 7 hours for 23 SNRs: -14 - 30
# If you get the error : Failed to get convolution algorithm, that probably means you're out of memory. Shut down all kernels and rerun the notebook.
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print("loss: {}".format(score[0]))
print("accuracy: {}".format(score[1]))
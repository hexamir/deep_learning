#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:34:59 2018

@author: victorchomel
"""

from mp1 import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
from keras import utils as np_utils
from keras.layers import UpSampling2D
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.layers import Flatten 

if False:
    [X_train, Y_train] = generate_dataset_classification(300, 20)
    
    y_train = np_utils.to_categorical(Y_train, 3)
    #print(y_train[:5])
    
    model = Sequential()
    nb_neurons = 5
    
    model.add(Dense(nb_neurons, input_shape=(5184,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(3, activation = 'sigmoid'))
    
    
    sgd = SGD(lr=0.01,
              decay=1e-6, momentum=0.9,
              nesterov=True)
    #model.compile(loss='mean_squared_error',
    #              optimizer = sgd)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer = sgd)
    #For adam : epochs = 50, converge plus lentement
    #For sgd : epochs = 10
    
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    
    
    
    X_test = generate_a_rectangle()
    X_test = X_test.reshape(1, X_test.shape[0])
    print(model.predict(X_test))
    
    print(model.get_weights()[0].shape)
    
    for i in range(3):
        plt.imshow(model.get_weights()[0][:, i].reshape((72,72)))
        plt.show()
    
    
if False :
    n = 2000
    [X_train, Y_train] = generate_dataset_classification(n, 20, True)
    new_X_train = X_train.reshape(n, 72, 72, 1)
    
    print(new_X_train.shape)
    
    y_train = np_utils.to_categorical(Y_train, 3)
        #print(y_train[:5])
        
    model = Sequential()
    nb_neurons = 5
    
    
    model.add(Conv2D(16, (5, 5), activation='relu',
              input_shape=(72, 72, 1)))
    
    
    
    model.add(MaxPooling2D(pool_size=(4, 4)))
    
    
    
    model.add(Flatten())
    
    model.add(Dense(32, activation='relu'))
    #
    model.add(Dropout(0.25))
    model.add(Dense(3, activation = 'sigmoid'))
    
    
    sgd = SGD(lr=0.01,
              decay=1e-6, momentum=0.9,
              nesterov=True)
    #model.compile(loss='mean_squared_error',
    #              optimizer = sgd)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer = sgd,
                  metrics = ['accuracy'])
    #For adam : epochs = 50, converge plus lentement
    #For sgd : epochs = 10
    
    model.fit(new_X_train, y_train, epochs=70, batch_size=32)
    
    
    [X_test, Y_test] = generate_test_set_classification()
    new_X_test = X_test.reshape(X_test.shape[0], 72, 72, 1)
    print(model.evaluate(new_X_test, Y_test))

if False:
    n = 1000
    [X_train, Y_train] = generate_dataset_regression(n, 20)
    
    new_Y_train = np.zeros((n, 6))
    c = 0
    for y in Y_train :
        
        new_y = []
        a = np.array(y)
        y = a.reshape((3,2))
        y = sorted(y, key = lambda a_entry : a_entry[0])
        for arr in y:
            new_y.append(arr[0])
            new_y.append(arr[1])
        new_Y_train[c] = np.array(new_y)
        
        c += 1
    
    new_X_train = X_train.reshape(n, 72, 72, 1)
    
    model = Sequential()
    nb_neurons = 5
    
    
    model.add(Conv2D(16, (5, 5), activation='relu',
              input_shape=(72, 72, 1)))
    
    model.add(Conv2D(32, (3, 3), activation='relu',
              input_shape=(72, 72, 1)))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    model.add(Flatten())
    
    model.add(Dense(64, activation='relu'))
    #
    model.add(Dropout(0.25))
    model.add(Dense(6))
    
    
    sgd = SGD(lr=0.01,
              decay=1e-6, momentum=0.9,
              nesterov=True)
    #model.compile(loss='mean_squared_error',
    #              optimizer = sgd)
    
    model.compile(loss='mean_squared_error',
                  optimizer = sgd,
                  metrics = ['accuracy'])
                  
    #For adam : epochs = 50, converge plus lentement
    #For sgd : epochs = 10
    
    model.fit(new_X_train, Y_train, epochs=100, batch_size=32)
    
    
    #visualize_prediction(X_train[0], Y_train[0])
    
    [X_test, Y_test] = generate_test_set_regression()
    
    #visualize_prediction(X_test[0], Y_test[0])
    
    for i in range(2):
        visualize_prediction(new_X_train[i], model.predict(new_X_train[i].reshape(1, 72, 72, 1)))
    
    for i in range(5):
        new_X_test = X_test.reshape(X_test.shape[0], 72, 72, 1)
        visualize_prediction(X_test[i], model.predict(new_X_test[i].reshape(1, 72, 72, 1)))
    
#%%
n = 500
[X_train, Y_train] = generate_dataset_denoising(n, noise=20.0)

new_X_train = X_train.reshape(n, 72, 72, 1)
new_Y_train = Y_train.reshape(n, 72, 72, 1)
#%%
seqmodel = Sequential()
seqmodel.add(Conv2D(32, (3, 3), padding='same', input_shape=(72, 72, 1)))
seqmodel.add(Activation('relu'))
seqmodel.add(MaxPooling2D((2, 2), padding='same'))

seqmodel.add(Conv2D(32, (3, 3), padding='same'))
seqmodel.add(Activation('relu'))
seqmodel.add(UpSampling2D((2, 2)))
seqmodel.add(Conv2D(1, (3, 3), padding='same'))
seqmodel.add(Activation('sigmoid'))

seqmodel.compile(optimizer='adadelta', loss='binary_crossentropy')

seqmodel.fit(new_X_train,
             new_Y_train,
             nb_epoch=50,
             batch_size=32,
             shuffle=True,
             validation_split=.20)

#%%    
x = new_X_train[0]
fig, ax = plt.subplots(figsize=(5, 5))
I = x.reshape((72,72))
ax.imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
ax.set_xlim([0,1])
ax.set_ylim([0,1])

y = new_Y_train[0]
fig, ax = plt.subplots(figsize=(5, 5))
I = y.reshape((72,72))
ax.imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
ax.set_xlim([0,1])
ax.set_ylim([0,1])



x_test = X_train[0].reshape(1, 72, 72, 1)
f1 = seqmodel.predict(x_test)

fig, ax = plt.subplots(figsize=(5, 5))
I = f1.reshape((72,72))
ax.imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
ax.set_xlim([0,1])
ax.set_ylim([0,1])

plt.show()
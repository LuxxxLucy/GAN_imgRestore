#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from random import sample
import os,random
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
import matplotlib.pyplot as plt
import seaborn as sns
import random, sys, keras
from keras.models import Model,model_from_json
from keras.utils import np_utils

WINDOW_WIDTH=int(32)
WINDOW_HEIGHT=int(32)

def build_model():
    opt=Adam(1e-3)
    g_input = Input(shape=[WINDOW_HEIGHT,WINDOW_WIDTH])
    H= Flatten()(g_input)
    H = Dense(10,init='zeros',activation="sigmoid")(H)
    H = Dense(WINDOW_WIDTH*WINDOW_HEIGHT,init='zeros',activation="sigmoid")(H)
    g_V = Reshape( (WINDOW_HEIGHT,WINDOW_WIDTH) )(H)
    generator = Model(g_input,g_V)
    generator.compile(loss='mean_squared_error', optimizer=opt)
    # generator.compile(loss='binary_crossentropy', optimizer=opt)
    generator.summary()
    return generator

def save_model(model,name="model"):
    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    generator.save_weights(name+".h5")

def load_model(name,opt = Adam(lr=1e-3),loss='mean_squared_error'):
    dropout_rate = 0.25
    dopt = Adam(lr=1e-3)

    # load json and create model
    json_file = open(name+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    generator = model_from_json(loaded_model_json)
    # load weights into new model
    generator.load_weights("generator.h5")
    generator.compile(loss=loss,optimizer=opt)
    return generator

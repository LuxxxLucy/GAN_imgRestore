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
from tqdm import tqdm

from utils import *
from model import *
from math import *
import cv2

WINDOW_WIDTH=int(32)
WINDOW_HEIGHT=int(32)


def train():
    return

def new_model():

    generator=build_model()

    for _ in range(20):
        true,noise = produce_data_bacth(200)
        generator.fit(noise,true,batch_size=32,epochs=1)
        plot_gen(generator)

    return

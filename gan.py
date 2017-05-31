#!/usr/bin/env python
#
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from pprint import pprint as pr

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


def build_generative_model():
    # Build Generative model ...
    nch = 3
    g_input = Input(shape=[WINDOW_HEIGHT,WINDOW_WIDTH,nch])
    H = Flatten()(g_input)
    # H = UpSampling2D(size=(2, 2))(H)
    # H = Convolution2D(int(nch/2), 3, 3, border_mode='same', init='glorot_uniform')(H)
    # H = BatchNormalization(mode=2)(H)
    # H = Activation('relu')(H)
    # H = Convolution2D(int(nch/4), 3, 3, border_mode='same', init='glorot_uniform')(H)
    # H = BatchNormalization(mode=2)(H)
    # H = Activation('relu')(H)
    # H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
    # H = Reshape( [ 1,28,28] )(H)
    # g_V = Activation('sigmoid')(H)
    H = Dense(nch*WINDOW_WIDTH*WINDOW_HEIGHT, kernel_initializer='glorot_uniform',activation="relu")(H)
    H = Dense(nch*WINDOW_WIDTH*WINDOW_HEIGHT,activation="sigmoid")(H)
    g_V = Reshape( (WINDOW_HEIGHT,WINDOW_WIDTH,nch) )(H)
    generator = Model(g_input,g_V)
    generator.compile(loss='binary_crossentropy', optimizer=opt)
    generator.summary()

    return generator

def buil_discriminative_model():
    # Build Discriminative model ...
    d_input = Input(shape=[WINDOW_HEIGHT,WINDOW_WIDTH,3])
    H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    H = Dense(256)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(2,activation='softmax')(H)
    discriminator = Model(d_input,d_V)
    discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
    discriminator.summary()
    return discriminator


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def plot_loss(losses):
#        display.clear_output(wait=True)
#        display.display(plt.gcf())
    plt.figure(figsize=(10,8))
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.show()

def plot_gen(generator,n_ex=16,dim=(4,4), figsize=(10,10) ):
    # show a generative image
    noise = np.random.uniform(0,1,size=[n_ex,100])
    data,noise = produce_data_bacth(4)
    generated_images = generator.predict(noise)

    plt.subplot(1,3,1),plt.imshow(data[0])#默认彩色，另一种彩色bgr
    plt.subplot(1,3,2),plt.imshow(noise[0])
    plt.subplot(1,3,3),plt.imshow(generated_images[0])
    plt.show()
    plt.savefig("./plot.png")


def plot_real(n_ex=16,dim=(4,4), figsize=(10,10) ):

    idx = np.random.randint(0,X_train.shape[0],n_ex)
    generated_images = X_train[idx,:,:,:]

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,0,:,:]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def train_for_n(generator,discriminator,GAN,nb_epoch=5000, plt_frq=25,BATCH_SIZE=32):
    pre_data=produce_data_bacth(20000)
    X_train=pre_data[0]

    ntrain = 1000
    trainidx = random.sample(range(0,X_train.shape[0]), ntrain)
    XT = X_train[trainidx,:,:,:]

    print("pre train the discriminator network")
    # Pre-train the discriminator network ...
    noise_gen = pre_data[1][:ntrain]
    generated_images = generator.predict(noise_gen)
    X = np.concatenate((XT, generated_images))
    n = XT.shape[0]
    y = np.zeros([2*n,2])
    y[:n,1] = 1
    y[n:,0] = 1

    make_trainable(discriminator,True)
    discriminator.fit(X,y, nb_epoch=15, batch_size=128)
    y_hat = discriminator.predict(X)

    # Measure accuracy of pre-trained discriminator network
    y_hat_idx = np.argmax(y_hat,axis=1)
    y_idx = np.argmax(y,axis=1)
    diff = y_idx-y_hat_idx
    n_tot = y.shape[0]
    n_rig = (diff==0).sum()
    acc = n_rig*100.0/n_tot
    print ("Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot))
    for e in tqdm(range(nb_epoch)):
        data_new = produce_data_bacth(BATCH_SIZE)

        # real image
        image_batch=data_new[0]
        # image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:,:,:]
        losses = {"d":[], "g":[]}
        # generative image
        noise_gen=data_new[1]
        # noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        generated_images = generator.predict(noise_gen)

        generator.fit(noise_gen,image_batch, nb_epoch=15, batch_size=128)

        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE,2])
        # one-hot code
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1

        make_trainable(discriminator,True)
        d_loss  = discriminator.train_on_batch(X,y)
        losses["d"].append(d_loss)

        # train Generator-Discriminator stack on input noise to non-generated output class
        # noise_tr = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        noise_tr = noise_gen
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:,1] = 1

        make_trainable(discriminator,False)
        #make_trainable(discriminator,False)
        g_loss = GAN.fit(noise_tr, y2 ,epochs=10)
        losses["g"].append(g_loss)

        # Updates plots
        if e%plt_frq==plt_frq-1:
            # plot_loss(losses)
            plot_gen()
            # serialize model to JSON
        save_model(generator,discriminator,GAN)


def save_model(generator,discriminator,GAN):
    model_json = generator.to_json()
    with open("generator.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    generator.save_weights("generator.h5")

    model_json = discriminator.to_json()
    with open("discriminator.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    discriminator.save_weights("discriminator.h5")

    model_json = GAN.to_json()
    with open("GAN.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    GAN.save_weights("GAN.h5")

def load_model():
    dropout_rate = 0.25
    opt = Adam(lr=1e-4)
    dopt = Adam(lr=1e-3)

    # load json and create model
    json_file = open('generator.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    generator = model_from_json(loaded_model_json)
    # load weights into new model
    generator.load_weights("generator.h5")
    generator.compile(loss='binary_crossentropy', optimizer=opt)

    # try:
    json_file = open('discriminator.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    discriminator = model_from_json(loaded_model_json)
    # load weights into new model
    discriminator.load_weights("discriminator.h5")
    discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)

    json_file = open('GAN.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    GAN = model_from_json(loaded_model_json)
    # load weights into new model
    GAN.load_weights("GAN.h5")
    GAN.compile(loss='categorical_crossentropy', optimizer=opt)

    return generator,discriminator,GAN

def plot_diff(i1,i2):
    plt.subplot(1,2,1),plt.imshow(i1)#默认彩色，另一种彩色bgr
    plt.subplot(1,2,2),plt.imshow(i2)
    plt.show()

def fake_image(img):
    '''
        add distortion to original image
    '''
    result=img.flatten()
    try:
        samples=np.random.choice(len(result),int(WINDOW_WIDTH*0.5)*int(WINDOW_HEIGHT*0.5),replace=True)
    except:
        pr(img)
        pr(result)

    for i in samples:
        result[i]=np.random.random_integers(256)
    result=np.reshape(result,(img.shape[0],img.shape[1],img.shape[2]))
    return result

def pad_to_window(img,shape1,shape2,value=0):
    padded = value * np.ones(shape=(shape1,shape2,img.shape[2]), dtype=img.dtype)
    padded[0:img.shape[0], 0:img.shape[1],:] = img
    return padded

def produce_data_bacth(batch_size=20):
    dirs = os.listdir("./data/")
    file_num=min(batch_size,len(dirs))
    images_names= sample(dirs,file_num)
    images= [ mpimg.imread("./data/"+temp) for temp in images_names ]
    count=0
    reals=[]
    fakes=[]
    for i in range(batch_size):
        # randomly choose a image
        j=np.random.random_integers(file_num-1)
        # randomly choose a location
        width=np.random.random_integers(int(images[j].shape[1]*0.8))
        height=np.random.random_integers(int(images[j].shape[0]*0.8))

        # get fake image
        images_data=images[j][:,:,:3]
        img=images_data[height:height+WINDOW_HEIGHT,width:width+WINDOW_WIDTH,:]
        img_f=fake_image(img)

        # paddding
        img=pad_to_window(img,WINDOW_HEIGHT,WINDOW_WIDTH)
        img_f=pad_to_window(img_f,WINDOW_HEIGHT,WINDOW_WIDTH)

        # append to the list
        reals.append(img)
        fakes.append(img_f)

    return (np.array(reals),np.array(fakes))


if __name__ == "__main__":

    img_rows, img_cols = 28, 28

    WINDOW_WIDTH=int(32)
    WINDOW_HEIGHT=int(32)

    dropout_rate = 0.25
    opt = Adam(lr=1e-4)
    dopt = Adam(lr=1e-3)

    generator,discriminator,GAN=load_model()
    data_new=produce_data_bacth(20000)
    image_batch=data_new[0]
    # image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:,:,:]
    losses = {"d":[], "g":[]}
    # generative image
    noise_gen=data_new[1]
    # noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
    generator.fit(image_batch,noise_gen,epochs=10)

    make_trainable(discriminator,False)
    train_for_n(generator,discriminator,GAN,nb_epoch=10, plt_frq=500,BATCH_SIZE=32)
    quit()

    print("start building the gan")
    generator=build_generative_model()
    discriminator=build_discriminative_model()
    # Freeze weights in the discriminator for stacked training


    # Build stacked GAN model
    gan_input = Input(shape=[WINDOW_HEIGHT,WINDOW_WIDTH,3])
    H = generator(gan_input)
    gan_V = discriminator(H)
    GAN = Model(gan_input, gan_V)
    GAN.compile(loss='categorical_crossentropy', optimizer=opt)
    GAN.summary()

    pre_data=produce_data_bacth(200)
    X_train=pre_data[0]

    ntrain = 100
    trainidx = random.sample(range(0,X_train.shape[0]), ntrain)
    XT = X_train[trainidx,:,:,:]

    print("pre train the discriminator network")
    # Pre-train the discriminator network ...
    noise_gen = pre_data[1][:ntrain]
    generated_images = generator.predict(noise_gen)
    X = np.concatenate((XT, generated_images))
    n = XT.shape[0]
    y = np.zeros([2*n,2])
    y[:n,1] = 1
    y[n:,0] = 1


    make_trainable(discriminator,True)
    discriminator.fit(X,y, nb_epoch=15, batch_size=128)
    y_hat = discriminator.predict(X)

    # Measure accuracy of pre-trained discriminator network
    y_hat_idx = np.argmax(y_hat,axis=1)
    y_idx = np.argmax(y,axis=1)
    diff = y_idx-y_hat_idx
    n_tot = y.shape[0]
    n_rig = (diff==0).sum()
    acc = n_rig*100.0/n_tot
    print ("Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot))

    # set up loss storage vector

    # Train for 6000 epochs at original learning rates
    train_for_n(generator,discriminator,GAN,nb_epoch=10, plt_frq=500,BATCH_SIZE=32)

    # Train for 2000 epochs at reduced learning rates
    opt.lr.set_value(1e-5)
    dopt.lr.set_value(1e-4)
    train_for_n(nb_epoch=2000, plt_frq=500,BATCH_SIZE=32)

    # Train for 2000 epochs at reduced learning rates
    opt.lr.set_value(1e-6)
    dopt.lr.set_value(1e-5)
    train_for_n(nb_epoch=200, plt_frq=500,BATCH_SIZE=32)

    # Plot the final loss curves
    # plot_loss(losses)

    # Plot some generated images from our GAN
    # plot_gen(25,(5,5),(12,12))


    # Plot real MNIST images for comparison
    # plot_real()

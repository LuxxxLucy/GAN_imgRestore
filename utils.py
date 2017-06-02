# -*- coding: UTF-8 -*-

import numpy as np

import os,random
from random import sample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from pprint import pprint as pr

WINDOW_WIDTH=int(32)
WINDOW_HEIGHT=int(32)

def plot_img(img):
    plt.imshow(img) # 显示图片
    plt.axis('off') # 不显示坐标轴
    plt.savefig("./plot_img.png")
    plt.show()

def plot_diff(i1,i2,name="plot_diff"):
    plt.subplot(1,2,1),plt.imshow(i1)#默认彩色，另一种彩色bgr
    plt.subplot(1,2,2),plt.imshow(i2)
    plt.savefig("./"+name+".png")
    plt.show()

def plot_loss(losses):
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
    true=data[0]
    generated=generated_images[0]

    plt.subplot(1,3,1),plt.imshow(true)#默认彩色，另一种彩色bgr
    plt.subplot(1,3,2),plt.imshow(noise[0])
    plt.subplot(1,3,3),plt.imshow(generated)
    plt.savefig("./plot_gen.png")
    print(true[0])
    print(noise[0][0])
    print(generated[0])
    plt.show()


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

def fake_image(img):
    '''
        add distortion to original image
    '''
    if(len(img.shape)==3):
        result=img.copy()
        noise_mask=np.zeros(shape=result.shape,dtype=np.uint8)

        for xi in range(noise_mask.shape[0]):
            ysamples=np.random.choice(result.shape[1],int(0.2*result.shape[1]))
            noise_mask[xi,ysamples,0]=1

            ysamples=np.random.choice(result.shape[1],int(0.6*result.shape[1]))
            noise_mask[xi,ysamples,1]=1

            ysamples=np.random.choice(result.shape[1],int(0.4*result.shape[1]))

            noise_mask[xi,ysamples,2]=1
        result=np.multiply(result, noise_mask)
        return result
    else:
        print("channel number error! in fake image producing")
        quit()

def pad_to_window(img,shape1,shape2,value=0):
    '''
    padding the image to a constant window size
    '''
    padded = value * np.ones(shape=(shape1,shape2,3), dtype=img.dtype)
    padded[0:img.shape[0], 0:img.shape[1]] = img
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
        # note that the default channel should be three
        images_data=images[j][:,:,:3]

        flag=0
        for i in images_data.flatten()[:10]:
            if i > 1:
                flag=1
        if flag==1 :
            continue

        img=images_data[height:height+WINDOW_HEIGHT,width:width+WINDOW_WIDTH,:]
        img_f=fake_image(img)

        # paddding
        img=pad_to_window(img,WINDOW_HEIGHT,WINDOW_WIDTH)
        img_f=pad_to_window(img_f,WINDOW_HEIGHT,WINDOW_WIDTH)

        # img=img[:,:,0]
        # img_f=img_f[:,:,0]


        # append to the list
        reals.append(img)
        fakes.append(img_f)

    return (np.array(reals),np.array(fakes))

# -*- coding: UTF-8 -*-
import os,sys
import cv2
import numpy as np
from math import *

from utils import *

WINDOW_WIDTH=int(32)
WINDOW_HEIGHT=int(32)
import numpy as np

from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片


def show_image(img,img_info="image"):
    print("img size")
    print(img.shape)
    plt.imshow(img) # 显示图片
    plt.axis('off') # 不显示坐标轴
    plt.show()
    print("show image")

def read_img(filename):
    # return cv3.imread(filename)
    try:
        return mpimg.imread(filename)[:,:,:3]
    except:
        image=mpimg.imread(filename)
        new_array=np.zeros(shape=(image.shape[0],image.shape[1],3))
        for i in range(3):
            new_array[:,:,i]+=image[:,:]
        return new_array

def bi_filtering(img):
    # t_img=np.float32(self.image)
    t_img=np.float32(img)
    result=np.uint8(cv2.bilateralFilter(t_img,9,75,75)*256)
    # result=np.uint8(cv2.bilateralFilter(t_img,20,75,75))
    return result

def distortion(img):
    return fake_image(img)

def interpolate(img):
    # result=img.flatten()
    result=np.copy(img)
    filtered=bi_filtering(img)
    for ci,i in enumerate(result):
        for cj,j in enumerate(result[ci]):
            try:
                for k in range(3):
                    pixel = result[ci][cj][k]
                    if(pixel==0 or pixel==256):
                        result[ci][cj][k]=filtered[ci][cj][k]
                        # result[ci][cj][k]=interpolate_with_context(img_ori)   [ci][cj][k]
            except:
                print("error")

    return filtered

def interpolate_with_context(img_ori,i,j):
    img=np.copy(img_ori)
    i_array=[ pixel for pixel in img[i-WINDOW_WIDTH:i+WINDOW_HEIGHT,j-WINDOW_HEIGHT:j+WINDOW_HEIGHT] ]
    result=sum(i_array)
    result=np.average(result, axis=0)

    return result

class Image:
    def __init__(self,filename):
        self.filename=filename
        try:
            self.image=read_img("./data_test/"+filename)
        except:
            print("sorry, file",filename,"does not exists. please check")
            quit()
        return

    def restore(self):
        # self.new_image=np.zeros(self.image.shape,np.uint8)
        # self.filtering()
        target_size=self.image.shape
        print(target_size)
        new_image=np.zeros(target_size,dtype=np.float32)
        # new_image+=self.image
        print(new_image.shape)
        for i in range(  int(ceil(target_size[0] / WINDOW_WIDTH)+1)):
            for j in range( int(ceil(target_size[1]/WINDOW_HEIGHT)+1)):
                source=self.image[i*WINDOW_WIDTH:(i+1)*WINDOW_WIDTH,j*WINDOW_HEIGHT:(j+1)*WINDOW_HEIGHT]
                previous_shape=source.shape
                source=pad_to_window(source,WINDOW_HEIGHT,WINDOW_WIDTH)
                # r1=restore_window(source)
                r1=restore_window_2(source)
                r1=r1[:previous_shape[0],:previous_shape[1],:]
                new_image[i*WINDOW_WIDTH:(i+1)*WINDOW_WIDTH,j*WINDOW_HEIGHT:(j+1)*WINDOW_HEIGHT,:]=r1[:,:,:]

                plot_diff(new_image,self.image,name="plot_diff_whole")

        new_image=new_image[:,:,::-1]
        cv2.imwrite("./data_test/result_skl/3140102299_"+self.filename,new_image*256)
        print ("restore! okay")

    def save_as(filename):
        pass

def restore_window(array):
    new_image=np.zeros(shape=array.shape,dtype=np.float32)
    for ch in range(3):
        channel=np.zeros(shape=array.shape[:3],dtype=np.float32)
        channel[:,:,ch]+=array[:,:,ch]
        X=[]
        X_no=[]
        Y=[]
        for i ,_ in enumerate(channel):
            for j,_ in enumerate(channel[i]):
                if(channel[i,j,ch]!=0):
                    X.append([i,j])
                    Y.append(channel[i][j][ch])
                else:
                    X_no.append([i,j])
        X=np.array(X)
        Y=np.array(Y)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        gp.fit(X, Y)
        result=gp.predict(np.array(X_no))
        for c,idx in enumerate(X_no):
            channel[idx[0],idx[1],ch]=result[c]
        new_image[:,:,ch]=channel[:,:,ch]
    print("new window okay")
    # plot_diff(new_image,array)

    return new_image


def restore_window_2(array):
    new_image=np.zeros(shape=array.shape,dtype=np.float32)
    channel=np.zeros(shape=array.shape[:3],dtype=np.float32)
    try:
        for ch in range(3):
            channel[:,:,ch]+=array[:,:,ch]
        X=[]
        X_no=[]
        Y=[]
        for i ,_ in enumerate(channel):
           for j,_ in enumerate(channel[i]):
               for ch in range(3):
                   if(channel[i,j,ch]!=0):
                       X.append([i,j,ch])
                       Y.append(channel[i][j][ch])
                   else:
                       X_no.append([i,j,ch])
        X=np.array(X)
        Y=np.array(Y)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        gp.fit(X, Y)
        result=gp.predict(np.array(X_no))


        for c,idx in enumerate(X_no):
           channel[idx[0],idx[1],idx[2]]=result[c]
        new_image[:,:,:]=channel[:,:,:]
        print("new window okay")
        # plot_diff(new_image,array)
        plot_diff(new_image,array)
    except:
        new_image=array
    return new_image


def restore_window_3(array):
    new_image=np.zeros(shape=array.shape,dtype=np.float32)
    channel=np.zeros(shape=array.shape[:3],dtype=np.float32)
    try:
        for ch in range(3):
            channel[:,:,ch]+=array[:,:,ch]
        X=[]
        X_no=[]
        Y=[]
        for i ,_ in enumerate(channel):
           for j,_ in enumerate(channel[i]):
               for ch in range(3):
                   if(channel[i,j,ch]!=0):
                       X.append([i,j,ch])
                       Y.append(channel[i][j][ch])
                   else:
                       X_no.append([i,j,ch])
        X=np.array(X)
        Y=np.array(Y)
        model = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression(fit_intercept=False))])
        # x = np.arange(5)
        # y = 3 - 2 * x + x ** 2 - x ** 3
        reg = model.fit(X, Y)

        # reg = linear_model.BayesianRidge()
        # reg.fit(X, Y)

        result=reg.predict(np.array(X_no))

        for c,idx in enumerate(X_no):
           channel[idx[0],idx[1],idx[2]]=result[c]
        new_image[:,:,:]=channel[:,:,:]
        print("new window okay")
        # plot_diff(new_image,array)
    except:
        print("error")
        new_image=array
    return new_image

if __name__ == "__main__":
    da,no=produce_data_bacth(20)
    print("start print")
    print(no.shape)
    for i in range(10):
        gen=restore_window(no[i])
        plt.subplot(1,3,1),plt.imshow(da[i])#默认彩色，另一种彩色bgr
        plt.subplot(1,3,2),plt.imshow(no[i])
        plt.subplot(1,3,3),plt.imshow(gen)
        print(gen[0][:2])
        print(no[i][0][:2])
        print(da[i][0][:2])
        plt.savefig("./plot_diff_test"+str(i)+".png")
        plt.show()
    for i in sys.argv[1:]:
        print(i)

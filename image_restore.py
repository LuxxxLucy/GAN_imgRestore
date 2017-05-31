# -*- coding: UTF-8 -*-
import os,sys
import cv2
import numpy as np
from math import *

from utils import *

WINDOW_WIDTH=int(200)
WINDOW_HEIGHT=int(200)

def show_image(img,img_info="image"):
    print("img size")
    print(img.shape)
    plt.imshow(img) # 显示图片
    plt.axis('off') # 不显示坐标轴
    plt.show()
    print("show image")

def read_img(filename):
    # return cv3.imread(filename)
    return mpimg.imread(filename)

def bi_filtering(img):
    # t_img=np.float32(self.image)
    t_img=np.float32(img)
    return cv2.bilateralFilter(t_img,9,75,75)

def distortion(img):
    return fake_image(img)


class Image:
    def __init__(self,filename):
        self.filename=filename
        try:
            self.image=read_img(filename)
        except:
            print("sorry, file",filename,"does not exists. please check")
            quit()
        return

    def restore(self):
        # self.new_image=np.zeros(self.image.shape,np.uint8)
        # self.filtering()
        target_size=self.image.shape
        print(target_size)
        new_image=np.zeros(target_size,dtype=np.uint8)
        print(new_image.shape)
        for i in range(ceil(target_size[0] / WINDOW_WIDTH)):
            for j in range(ceil(target_size[1]/WINDOW_HEIGHT)):
                source=self.image[i*WINDOW_WIDTH:(i+1)*WINDOW_WIDTH,j*WINDOW_HEIGHT:(j+1)*WINDOW_HEIGHT]
                r1=self.restore_window(source)
                # display(source,r1)
                new_image[i*WINDOW_WIDTH:(i+1)*WINDOW_WIDTH,j*WINDOW_HEIGHT:(j+1)*WINDOW_HEIGHT,:]=r1[:,:,:]

        plot_diff(self.image,new_image)
        print ("restore! okay")

    def save_as(filename):
        pass

    def restore_window(self,array):
        # result=np.zeros(shape=(WINDOW_WIDTH,WINDOW_HEIGHT),np.uint8)
        # result = bi_filtering(array)
        result=distortion(array)
        return result

if __name__ == "__main__":
    for i in sys.argv[1:]:
        print(i)

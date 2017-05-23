import os,sys
import cv2
import numpy as np

from img_utility import *


class Image:
    def __init__(self,filename):
        self.filename=filename
        try:
            self.image=read_img(filename)
            show_image(self.image)
        except:
            print("sorry, file",filename,"does not exists. please check")
            quit()
        return

    def restore(self):
        self.new_image=np.zeros(self.image.shape,np.uint8)
        show_image(self.new_image)
        print ("restore!")

    def save_as(filename):
        pass

if __name__ == "__main__":
    for i in sys.argv[1:]:
        print(i)

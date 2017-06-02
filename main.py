# -*- coding: UTF-8 -*-
from image_restore import *
import os,sys

if __name__ == "__main__":
    for i in os.listdir("./data_test"):
        try:
            if(i!=".DS_Store"):
                print(i)
                i = Image(i)
                i.restore()
        except:
            continue

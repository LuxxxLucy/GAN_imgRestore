# -*- coding: UTF-8 -*-
from image_restore import *
import os,sys
# from gan import *


if __name__ == "__main__":
    for i in os.listdir("./data_test"):
        if(i!=".DS_Store"):
            print(i)
            i = Image(i)
            i.restore()

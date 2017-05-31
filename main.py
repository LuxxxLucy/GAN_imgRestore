# -*- coding: UTF-8 -*-
from image_restore import Image
import os,sys
# from gan import *


if __name__ == "__main__":
    for i in sys.argv[1:]:
        i = Image(i)
        i.restore()
        print(i)

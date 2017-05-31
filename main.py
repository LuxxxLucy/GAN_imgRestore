from image_restore import Image
import os,sys
from gan import *

if __name__ == "__main__":

    generator,_,_=load_model()
    plot_gen(generator)

    for i in sys.argv[1:]:
        i = Image(i)
        i.restore()
        print(i)

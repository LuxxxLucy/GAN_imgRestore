import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def show_image(img,img_info="image"):
    # cv2.namedWindow(img_info)
    # cv2.imshow(img_info,img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("show image")
    plt.imshow(img) # 显示图片
    plt.axis('off') # 不显示坐标轴
    plt.show()
    return

def read_img(filename):
    return mpimg.imread(filename)

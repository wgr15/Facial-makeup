#使用seamlessClone函数进行图像融合

import cv2
import numpy as np 


img_des = cv2.imread('./images/1.jpg')
img_src = cv2.imread('./images/1_left.png')

mask = 255 * np.ones((img_src.shape[0] , img_src.shape[1] , 3) , img_src.dtype)

center = (int(img_des.shape[1]/2)  , int(img_des.shape[0]/2))

#五个参数
#img_src : 原图像
#img_des ：要插入的图像
#mask为希望插入的图片位置，通道数为3 形状与src相同，要插的位置为(255,255,255),不需要插值的位置为0
#center :插入的图片中心位置在目标图像中的位置，元组类型
#cv.NORMAL_CLONE ： 融合的方式，一般是这个
output = cv2.seamlessClone(img_src , img_des , mask , center , cv2.NORMAL_CLONE)

cv2.imwrite('./clone.png' , output)
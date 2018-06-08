import cv2
import dlib
import numpy as np 
import sys


#创建一个人脸检测器类。
face_detector = dlib.get_frontal_face_detector()

#官方提供的模型构建特征提取器，提前训练好的，可以直接进行人脸关键点的检测。
landmark_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


img = cv2.imread('./images/' + sys.argv[1])
img_processed = img.copy()


#使用检测器对输入的图片进行人脸检测,返回矩形坐标
facesrect = face_detector(img)

pos = []
for k , d in enumerate(facesrect):
	shape = landmark_predictor(img , d)
	for i in range(17 , 27):
		pos.append((shape.part(i).x , shape.part(i).y))

for i in range(10):
	img_processed = cv2.circle(img_processed , pos[i] , 2 , (0 , 0 , 255))


#cv2.imshow("test" , img_processed)
cv2.imwrite('./images/'+ sys.argv[2] +'_processed.jpg' , img_processed)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

2_left = [(206 , 105) , (181 , 90) , (149 ,86) ,(112, 86) , (90 , 101)] 
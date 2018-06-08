import cv2
import dlib
import numpy as np 
import sys


#创建一个人脸检测器类。
face_detector = dlib.get_frontal_face_detector()

#官方提供的模型构建特征提取器，提前训练好的，可以直接进行人脸关键点的检测。
landmark_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


img = cv2.imread('./images/5.jpg' )
left = cv2.imread('./images/1_left.png')


img_processed = img.copy()


#使用检测器对输入的图片进行人脸检测,返回矩形坐标
facesrect = face_detector(img)

pos = []
for k , d in enumerate(facesrect):
	shape = landmark_predictor(img , d)
	for i in range(17 , 27):
		pos.append((shape.part(i).x , shape.part(i).y))

pos = np.array(pos)
''''
left_points = []
first_point = np.array([35 , 129])
delta = np.array(first_point[0] - pos[0,0] , first_point[1] - pos[0,1])

left_points = pos + delta
'''
left_points = np.array([(35 , 129) , (81 , 88), (134 , 74) , (202 , 84) , (261 , 106)])

#left_points = np.array([(89 , 100) , (111 , 85) , (148 , 85) , (180 , 89) , (205 , 104)])

H,mask = cv2.findHomography(left_points , pos[0:5] , cv2.RANSAC )


x_max = int(np.dot([299 , 199 , 1] , H[0]))
y_max = int(np.dot([299 , 199 , 1] , H[1]))


img_out = cv2.warpPerspective(left , H , (x_max , y_max))

for i in range(x_max):
	for j in range(y_max):
		if (img_out[j , i] < [50,50,50]).all():
			pass
		else:
			img_processed[j , i] = (img_out[j , i])




cv2.imwrite("./images/final.png" , img_processed)
cv2.imwrite("./images/out.png" , img_out)
#cv2.waitKey()
#cv2.destroyAllWindows()

#[(39 , 129) , (90 , 85), (136 , 75) , (202 , 80) , (261 , 102)]




'''

#cv2.imshow("test" , img_processed)
cv2.imwrite('./images/'+ sys.argv[2] +'_processed.jpg' , img_processed)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

[(206 , 105) , (181 , 90) , (149 ,86) ,(112, 86) , (90 , 101)] 

left = np.array([(89 , 100) , (111 , 85) , (148 , 85) , (180 , 89) , (205 , 104)])
'''
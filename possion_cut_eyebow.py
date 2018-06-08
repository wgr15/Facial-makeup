import cv2
import dlib
import numpy as np 
import sys


#创建一个人脸检测器类。
face_detector = dlib.get_frontal_face_detector()

#官方提供的模型构建特征提取器，提前训练好的，可以直接进行人脸关键点的检测。  
#18-22为左眉毛      23-27为右眉毛  37-40为左眼皮    43-46为右眼皮

landmark_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


img = cv2.imread('./images/10.jpeg' )
left = cv2.imread('./images/1_left.png')


img_processed = img.copy()


#使用检测器对输入的图片进行人脸检测,返回矩形坐标
facesrect = face_detector(img)


left_eye_bow = []
left_eye = []
right_eye_bow = []
right_eye = []
x_sum = 0
y_sum = 0

for k , d in enumerate(facesrect):
	shape = landmark_predictor(img , d)
	for i in range(17 , 22):
		left_eye_bow.append((shape.part(i).x , shape.part(i).y -20))
		x_sum += shape.part(i).x
		y_sum += shape.part(i).y

	for i in range(22 , 27):
		right_eye_bow.append((shape.part(i).x , shape.part(i).y))
	flag = 0
	for i in range(36 ,  40):
		if flag <= 2:
			left_eye.append(( int(shape.part(i).x * 0.6 + left_eye_bow[flag][0] * 0.4) , int(shape.part(i).y * 0.6 + left_eye_bow[flag][1] * 0.4)))
		else:
			left_eye.append(( int(shape.part(i).x * 0.6 + left_eye_bow[flag+1][0] * 0.4) , int(shape.part(i).y * 0.6 + left_eye_bow[flag+1][1] * 0.4)))
		x_sum += left_eye[flag][0]
		y_sum += left_eye[flag][1]
		flag += 1

	for i in range(42 , 46):
		right_eye.append((shape.part(i).x , shape.part(i).y))

center = (int(x_sum/9) , int(y_sum/9))



left_eye_copy = left_eye.copy()
left_eye_copy.reverse()
ploy = left_eye_bow + left_eye_copy


left_eye_bow = np.array(left_eye_bow)
left_eye = np.array(left_eye)
right_eye_bow = np.array(right_eye_bow)
right_eye = np.array(right_eye)


ploy = np.array(ploy)


mask_left = np.zeros(img.shape , img.dtype)
skin_mask_left = np.zeros(img.shape , img.dtype)


skin_color = img[left_eye_bow[0 , 0] , left_eye_bow[0 , 1] - 10]

cv2.fillPoly(mask_left , [ploy] , (255,255,255))
cv2.fillPoly(skin_mask_left , [ploy] , skin_color.tolist())

#cv2.imwrite("./skin_mask_left.png" , skin_mask_left)
#cv2.imwrite("./skin_mask_left.png" , )
output = cv2.seamlessClone(skin_mask_left , img , mask_left , center , cv2.NORMAL_CLONE)
#cv2.imwrite("./mask_left.png" , mask_left)



cv2.imwrite("./cut_eyebow.png" , output)
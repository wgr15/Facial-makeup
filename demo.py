import cv2
import dlib
import numpy as np


#创建一个摄像头捕捉类
cap = cv2.VideoCapture(0)


#创建一个人脸检测器类。
face_detector = dlib.get_frontal_face_detector()

#官方提供的模型构建特征提取器，提前训练好的，可以直接进行人脸关键点的检测。
landmark_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

#将opencv默认的BGR转换为RGB。
#rgb_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

while(1):

	#从摄像头捕捉一张图片
	ret , frame = cap.read()

	####################################
	#我们需要写的代码，frame为读入的图片，三个通道分别为bgr
	#
	#
	####################################

	#使用检测器对输入的图片进行人脸检测,返回矩形坐标
	facesrect = face_detector(frame)

	#如果存在人脸，就进行特征点检测并标记
	if len(facesrect) > 0:
		
		#使用特征提取器对图片中人脸部位进行检测，每一个人脸检测到68个点，其中17到26一共10个点为左右眉毛所在的位置
		for k ,d in enumerate(facesrect):
			shape = landmark_predictor(frame , d)
			for i in range(17 , 27):
				if i != 21 and i != 26:
					frame = cv2.line(frame , (shape.part(i).x  , shape.part(i).y) , (shape.part(i+1).x , shape.part(i+1).y ), (0,0,0) , 5)
				else:
					pass
				
	#将图片显示出来
	cv2.imshow("capture" , frame)
	
	#检测按键，如果是ESC键则退出，否则继续执行
	k = cv2.waitKey(1) 
	if k == 27:
		break


#释放摄像头
cap.release()

#关闭所有窗口
cv2.destroyAllWindows()
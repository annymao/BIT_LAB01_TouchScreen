from __future__ import print_function
import cv2
import numpy as np
import time
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC

import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from scipy.spatial import distance


def FindCenter(contours):
	sum = np.zeros((2,))
	for c in contours:
		sum += c[0]
	sum /= contours.shape[0]
	return sum.astype(int)
	
# 選擇攝影機
cap = cv2.VideoCapture(1)
# initialize the handwrite frame
handWrite = np.zeros((480,640,3), np.uint8)

#initialize the previous centriod (use previousCentroid and centroid to draw line)
previousCentroid =np.array([-1,-1])

#model = load_model('models/my_model2.h5')
# initialize startTime 

timeThresh = 2.0
velThresh = 80.0
disThersh = 20.0

startTime = time.time()
longPressTime = time.time()
tapCount = 0
frameCount = 0
status = "IDLE"
longPressed = False
isScroll = False
while(True):
	# 從攝影機擷取一張影像
	ret, frame = cap.read()
	# 顯示圖片
	## Flip the image and change to grayscale
	flipImage = cv2.flip(frame,1)
	gscale = cv2.cvtColor(flipImage,cv2.COLOR_BGR2GRAY)

	# binarize
	ret,thresh = cv2.threshold(gscale,200,255,cv2.THRESH_TOZERO)
	#cv2 findContours
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	newContours=[]

	# only take contour size > 300
	for c in contours:
		if(c.shape[0]>300):
			newContours.append(c)

	# draw on the handwrite frame
	if(len(newContours)>0):
		starttime = time.time()
		centroid = FindCenter(newContours[0])
		cv2.drawMarker(flipImage, (centroid[0],centroid[1]), (0,0,255), cv2.MARKER_CROSS, 10, 2, 8)
		cv2.drawContours(flipImage,newContours,-1,(0,0,255),3)
		## Draw the handwrite output
		# if(previousCentroid[0] == -1):
		# 	previousCentroid = centroid
		# else:
		# 	cv2.line(handWrite,  (previousCentroid[0],previousCentroid[1]),  (centroid[0],centroid[1]), (255, 255, 255), 14)
		# 	previousCentroid = centroid
		cv2.circle(handWrite,(centroid[0],centroid[1]), 10, (255, 255, 255), -1)
	else:
		# reset centroid 
		#previousCentroid=np.array([-1,-1])
		centroid = np.array([-1,-1])
		# if((time.time() - starttime)>3.0):
		# 	handWrite.fill(0)
		# 	starttime = time.time()

	if(status == 'IDLE'):
		if(centroid[0]!=-1):
			status = 'BEGIN'
			if(tapCount == 0):
				startTime = time.time()
			longPressTime = time.time()
			previousCentroid = centroid
		if tapCount>=1 and (time.time()-startTime) > timeThresh:
			tapCount = 0
			print("-----TAP-----")

	elif(status == 'BEGIN'):
		if(longPressed!=True and (time.time()-longPressTime)>timeThresh):
			print("-----LONG PRESS-----")
			longPressed = True
		if(centroid[0] == -1):
			status = 'END'
		elif(distance.euclidean(previousCentroid, centroid)>disThersh):
			frameCount=0
			status = 'MOVE'
			isScroll = False
		previousCentroid = centroid
	elif(status == 'END'):
		if(longPressed):
			longPressed = False
			tapCount=0
		elif(tapCount == 0):
			tapCount+=1
		else:
			print("-----MULTI TAP-----")
			tapCount = 0
		handWrite.fill(0)
		status = 'IDLE'
	elif(status == 'MOVE'):
		frameCount+=1
		if(frameCount>=2):
			print("Distance: ",distance.euclidean(previousCentroid, centroid))
			if( isScroll == False and centroid[0]==-1 ):#and distance.euclidean(previousCentroid, centroid)>velThresh):
				print("-----SWIPE-----")
			else:
				previousCentroid = centroid
				print("-----SCROLL-----")
				isScroll = True
		if(centroid[0] == -1):
			handWrite.fill(0)
			tapCount=0
			status = 'IDLE'

	else:
		status = 'IDLE'



	cv2.imshow('frame', flipImage)
	cv2.imshow('HandWriteFrame', handWrite)

	# gscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


	# 若按下 q 鍵則離開迴圈
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()


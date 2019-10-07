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


def FindCenter(contours):
	sum = np.zeros((2,))
	for c in contours:
		sum += c[0]
	sum /= contours.shape[0]
	return sum.astype(int)
	

def prepareInput(im):
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
	#im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
	ret, im_th = cv2.threshold(im_gray,127,255,cv2.THRESH_TOZERO)
	#get thershold image( maybe not used )
	# ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Get rectangles contains each contour
	# rects = [cv2.boundingRect(ctr) for ctr in ctrs]
	# for rect in rects:
	# 	print(rect[2]*rect[3])


	test = cv2.resize(im_th[:,:], (28, 28), interpolation=cv2.INTER_AREA)
	test = np.reshape(test, (1,28,28,1)).astype(float)
	return test

def classifier(im):
    # Load the classifier
	clf = joblib.load("digits_cls2.pkl")

	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
	ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
	#get thershold image( maybe not used )
	roi = cv2.resize(im_th[:,:], (28, 28), interpolation=cv2.INTER_AREA)	
	roi = cv2.dilate(roi, (3, 3))
	# Calculate the HOG features
	roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
	nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
	print("recognized: ",int(nbr[0]))
	#cv2.putText(im, str(int(nbr[0])), (im.shape[0[,14),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
	#cv2.imshow("Resulting Image with Rectangular ROIs", im)

# 選擇攝影機
cap = cv2.VideoCapture(1)
# initialize the handwrite frame
handWrite = np.zeros((480,640,3), np.uint8)

#initialize the previous centriod (use previousCentroid and centroid to draw line)
previousCentroid =np.array([-1,-1])

model = load_model('models/my_model2.h5')
# initialize startTime 
starttime = 0

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
		if(previousCentroid[0] == -1):
			previousCentroid = centroid
		else:
			cv2.line(handWrite,  (previousCentroid[0],previousCentroid[1]),  (centroid[0],centroid[1]), (255, 255, 255), 14)
			previousCentroid = centroid
		cv2.circle(handWrite,(centroid[0],centroid[1]), 10, (255, 255, 255), -1)
	else:
		# reset centroid 
		previousCentroid=np.array([-1,-1])
		# start recognize if timeElapsed > 3 s
		if((time.time() - starttime)>3.0):
			#classifier(handWrite)
			x = prepareInput(handWrite)
			if(len(np.unique(handWrite))==1):
				print("Wait")
			else:
				out = model.predict(x)
				print(np.argmax(out))
			handWrite.fill(0)
			starttime = time.time()
			

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


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('A4.jpg')
row,col,p = img.shape

print(img.shape)

img = cv2.resize(img,(int(col/4),int(row/4)))
row,col,p = img.shape

print(type(img))

b,g,r = cv2.split(img)

print(type(b))

img1 = cv2.merge((b,g,r))

print(type(img1))

def nothing(x):
	pass

cv2.namedWindow('track',cv2.WINDOW_NORMAL)
cv2.namedWindow('track_',cv2.WINDOW_NORMAL)

cv2.createTrackbar('bt','track',0,255,nothing)
cv2.createTrackbar('gt','track',0,255,nothing)
cv2.createTrackbar('rt','track',0,255,nothing)
cv2.createTrackbar('lh','track_',0,180,nothing)
cv2.createTrackbar('uh','track_',0,180,nothing)
cv2.createTrackbar('ls','track_',0,255,nothing)
cv2.createTrackbar('us','track_',0,255,nothing)
cv2.createTrackbar('lv','track_',0,255,nothing)
cv2.createTrackbar('uv','track_',0,255,nothing)

while(True):

	bt = cv2.getTrackbarPos('bt','track')
	gt = cv2.getTrackbarPos('gt','track')
	rt = cv2.getTrackbarPos('rt','track')

	lh = cv2.getTrackbarPos('lh','track_')
	uh = cv2.getTrackbarPos('uh','track_')
	ls = cv2.getTrackbarPos('ls','track_')
	us = cv2.getTrackbarPos('us','track_')
	lv = cv2.getTrackbarPos('lv','track_')
	uv = cv2.getTrackbarPos('uv','track_')

	'''
	b_ = b*(10/100)
	#b_ = b_.astype(int)
	g_ = g*(100/100)
	#g_ = g_.astype(np.ndarray)
	r_ = r*1

	cv2.imshow('blue',b_)
	cv2.imshow('gren',g_)
	cv2.imshow('red',r_)

	mg = cv2.merge((b_,g_,r_))
	

	img_ = np.zeros((row,col,3),np.uint8)
	print(img_.shape)

	for i in range(row):
		for j in range(col):

			img_[i,j,0] = int(img[i,j,0]*bt/100)
			img_[i,j,1] = int(img[i,j,1]*gt/100)
			img_[i,j,2] = int(img[i,j,2]*rt/100)
	'''

	b_ = b-bt
	g_ = g-gt
	r_ = r+rt

	img_ = cv2.merge((b_,g_,r_))

	hsv = cv2.cvtColor(img_,cv2.COLOR_BGR2HSV)
	low = np.array([lh,ls,lv], np.uint8)
	high = np.array([uh,us,uv], np.uint8)

	mask = cv2.inRange(hsv,low,high)

	cv2.imshow('mask',mask)

	cv2.imshow('cvt',img_)

	if cv2.waitKey(30) & 0xFF == 27:
		break

cv2.waitKey(0)
cv2.destroyAllWindows()

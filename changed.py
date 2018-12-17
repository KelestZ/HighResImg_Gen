import cv2
import os
import numpy as np
dir_='/home/nfs/zpy/xrays/HighResImg_Gen/generations2/images/'
dir2_='/home/nfs/zpy/xrays/HighResImg_Gen/generations2/images2/'
li = os.listdir(dir_)


for i in li:
	a=cv2.imread(dir_+i, -1)
	b = np.zeros([128, 128, 4])
	b[:,:,0]= a
	b[:, :, 1] = a
	b[:, :, 2] = a
	b[:, :, -1] = 255
	b = b.astype(np.uint8)
	cv2.imwrite(dir2_+i,b)


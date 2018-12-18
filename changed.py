import cv2
import os
import numpy as np
dir_='/home/nfs/zpy/xrays/generations_1217_8w_istrue/images/'
dir2_='/home/nfs/zpy/xrays/generations_1217_8w_istrue/images2/images/'
li = os.listdir(dir_)


for i in li:
	a=cv2.imread(dir_+i, -1)

	c=cv2.imread(dir2_+i, -1)

	print(a[0,:10],c[0,:10])


	#b = np.zeros([128, 128])
	#b[:, :]= a.astype(np.float32)

	#print(a[0,:10],b[0,:10],type(b))
	# print(np.max(a),np.min(a),np.max(b),np.min(b),a[0,0],b[0,0])
	#cv2.imwrite(dir2_+i, b)


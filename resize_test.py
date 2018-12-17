import numpy as np
import cv2
import os
from sklearn.metrics import mean_squared_error
test_dir = '/home/nfs/zpy/xray_images/train_images_128x128/'

img_dir ='/home/nfs/zpy/xrays/HighResImg_Gen/generations3/images/'

sum = 0
ct =0
for filename in os.listdir(img_dir):
	if filename.endswith('.png'):
		print(filename)
		absolute_filename = os.path.join(test_dir, filename)
		gen_dir = img_dir+ filename

		image = cv2.imread(absolute_filename, cv2.IMREAD_UNCHANGED)

		resized_image = image[:,:,0]#cv2.resize(image, (128, 128))[:,:,0]

		img_gen = cv2.imread(gen_dir, cv2.IMREAD_UNCHANGED)
		resized_image=resized_image.reshape([-1])
		img_gen=img_gen.reshape([-1])
		a = mean_squared_error(resized_image,img_gen)
		print(resized_image[:10], img_gen[:10])
		print(ct, 'MSE:', a)
		sum+=a
		ct+=1

print('sum',sum)

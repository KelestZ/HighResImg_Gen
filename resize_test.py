import numpy as np
import cv2
import os
from sklearn.metrics import mean_squared_error
test_dir = '/home/nfs/zpy/xray_images/test_images_64x64/'

img_dir ='/home/nfs/zpy/xrays/generations_1217_8w_istrue/images/' #'/home/nfs/zpy/xrays/HighResImg_Gen/generations/images/'#'/home/nfs/zpy/xrays/generations_1217_2/images/'
new_dir='/home/nfs/zpy/xrays/generations_1217_8w_istrue/images2/'
sum = 0.0
ct =0

sum2 = 0.0
def cal_mse(a,b):
	c=a-b
	print(a[0, :10], b[0, :10], c[0, :10])
	return np.sqrt(np.mean(np.square(a-b)))


for filename in os.listdir(img_dir):
	if filename.endswith('.png'):
		print(filename)
		absolute_filename = os.path.join(test_dir, filename)
		gen_dir = img_dir+ filename
		#print(absolute_filename)
		image = cv2.imread(absolute_filename, 0)# cv2.IMREAD_UNCHANGED)
		resized_image = cv2.resize(image, (128, 128)).astype(np.float32)

		img_gen = cv2.imread(gen_dir, cv2.IMREAD_UNCHANGED).astype(np.float32)

		print(resized_image[0,:10], img_gen[0,:10])
		mean = np.mean(img_gen.astype(np.float32)-resized_image.astype(np.float32))

		img_gen -= mean
		#cv2.imwrite(new_dir+filename,img_gen)

		#resized_image=resized_image.reshape(-1)
		#img_gen=img_gen.reshape(-1)
		a = np.sqrt(mean_squared_error(resized_image.astype(np.float32), img_gen.astype(np.float32)))
		b = cal_mse(resized_image.astype(np.float32), img_gen.astype(np.float32)) #.astype(np.int32)
		#print(a, b, mean)
		sum+=a
		sum2+=b
		ct+=1

print('sum',sum,sum2,sum2/ct)

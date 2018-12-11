import os
import tensorflow as tf
import numpy as np

from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import cv2

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

class ImageDataGenerator(object):
    def __init__(self, file_path='/Users/zpy/Desktop/xray_images/', train_data_dic='train_images_64x64/',
                 train_label_dic='train_images_128x128/',batch_size =32,
                 shuffle=True, img_size=[64, 64],label_size=[128, 128], buffer_size=1000):
        '''

            Args:
            txt_file: Path to the text file.
            mode: Either 'train' or 'test'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
            Raises:
                ValueError: If an invalid mode is passed.
        '''

        self.file_path = file_path
        self.train_data_dic = train_data_dic
        self.train_label_dic = train_label_dic
        # self.num_classes = num_classes

        # retrieve the data from the dictionary
        self._load_data()

        # number of samples in the dataset
        self.data_size = len(self.labels)

        self.img_size = img_size  # (64, 64)
        self.label_size = label_size  # (128, 128)
        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor (build in the system)
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels_paths = convert_to_tensor(self.labels, dtype=dtypes.string) # it is a path, too.

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        #if mode == 'train':
        data = data.map(map_func=self._parse_function_train, num_parallel_calls=8)

        data = data.prefetch(buffer_size)
        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)
        self.data = data

    def _load_data(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []

        path = self.file_path+self.train_data_dic
        for i in os.listdir(path):
            self.img_paths.append(self.file_path + self.train_data_dic + i)
            self.labels.append(self.file_path + self.train_label_dic + i)


    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels_paths
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels_paths = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, image_path, label_path):
        """Input parser for samples of the training set."""

        # load and preprocess the image
        img_string = tf.read_file(image_path)
        print(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=0)
        #img_resized = tf.image.resize_images(img_decoded, [self.img_size[0], self.img_size[1]], method=0) # 0:bilinear

        #img_centered = tf.subtract(img_resized, IMAGENET_MEAN)
        # RGB -> BGR
        #img_bgr = img_centered[:, :, ::-1]


        # convert label to ground truth png
        img_string2 = tf.read_file(label_path)
        img_decoded2 = tf.image.decode_png(img_string2)
        #img_gt = tf.image.resize_images(img_decoded2, [self.label_size[0], self.label_size[1]], method=1) # 1:Nearest

        return img_decoded, img_decoded2 #img_bgr, img_gt


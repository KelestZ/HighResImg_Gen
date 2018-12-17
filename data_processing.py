import os
import tensorflow as tf
import numpy as np
import collections
from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from lib.ops import *
import scipy.misc as sic
import cv2

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
# Define the dataloader

def data_loader(FLAGS):
    with tf.device('/cpu:0'):
        # Define the returned data batches
        Data = collections.namedtuple('Data', ' inputs, targets, image_count, steps_per_epoch')
        tr_data = ImageDataGenerator(FLAGS.train_file,
                                     img_size=[FLAGS.DATA_HEIGHT, FLAGS.DATA_WIDTH],
                                     label_size=[FLAGS.LABEL_HEIGHT, FLAGS.LABEL_WIDTH],
                                     batch_size=FLAGS.batch_size,
                                     shuffle=True,
                                     FLAGS=FLAGS)

        #steps_per_epoch = int(np.floor(tr_data.data_size / FLAGS.batch_size))
        steps_per_epoch = tr_data.data_size // FLAGS.batch_size
        input_images = tr_data.input_images
        target_images = tr_data.target_images

        if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
            input_images.set_shape([FLAGS.DATA_HEIGHT, FLAGS.DATA_WIDTH, 1])
            target_images.set_shape([FLAGS.LABEL_HEIGHT, FLAGS.LABEL_WIDTH, 1])

        if FLAGS.mode == 'train':
            inputs_batch, targets_batch = tf.train.shuffle_batch([input_images, target_images],
                batch_size=FLAGS.batch_size, capacity=FLAGS.image_queue_capacity + 4 * FLAGS.batch_size,
                min_after_dequeue=FLAGS.image_queue_capacity, num_threads=FLAGS.queue_thread)
        else:
            inputs_batch, targets_batch = tf.train.batch([input_images, target_images],
                batch_size=FLAGS.batch_size, num_threads=FLAGS.queue_thread, allow_smaller_final_batch=True)


        if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
            inputs_batch.set_shape([FLAGS.batch_size, FLAGS.DATA_HEIGHT, FLAGS.DATA_WIDTH, 1])
            targets_batch.set_shape([FLAGS.batch_size, FLAGS.LABEL_HEIGHT, FLAGS.LABEL_WIDTH, 1])


    return Data(
        inputs=inputs_batch,
        targets=targets_batch,
        image_count=tr_data.data_size,
        steps_per_epoch=steps_per_epoch)


def inference_data_loader(FLAGS):
    with tf.device('/cpu:0'):
        # Define the returned data batches
        Data = collections.namedtuple('Data', ' inputs, size, paths_LR')
        test_data = Inference_DataGenerator(FLAGS.train_file,
                                     train_data_dic='test_images_64x64/',
                                     img_size=[FLAGS.DATA_HEIGHT, FLAGS.DATA_WIDTH],
                                     shuffle=False,
                                     FLAGS=FLAGS)

    return Data(
        inputs=test_data.input_images,
        size=test_data.data_size,
        paths_LR=test_data.img_paths)



class Inference_DataGenerator(object):
    def __init__(self, file_path='/home/nfs/zpy/xrays/HighResImg_Gen/xray_images/', train_data_dic='train_images_64x64/',
                 shuffle=False, img_size=[64, 64], buffer_size=1000, FLAGS=None):
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

        # retrieve the data from the dictionary
        self._load_data()

        # number of samples in the dataset
        self.img_size = img_size  # (64, 64)
        self.data_size = len(self.img_paths)

        # Read in and preprocess the images
        input_image_LR = [self._preprocess_test(_) for _ in self.img_paths]
        self.input_images = input_image_LR


    def _preprocess_test(self, name):
        img = cv2.imread(name, 0).astype(np.float32)
        img = np.expand_dims(img, -1)
        img = img/np.max(img)
        return img

    def _load_data(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        path = self.file_path+self.train_data_dic
        print('path', path)
        for i in os.listdir(path):
            self.img_paths.append(self.file_path + self.train_data_dic + i)
            self.img_paths.append(self.file_path + self.train_data_dic + i)


class ImageDataGenerator(object):
    def __init__(self, file_path='/home/nfs/zpy/xrays/HighResImg_Gen/xray_images/', train_data_dic='train_images_64x64/',
                 train_label_dic='train_images_128x128/',batch_size =32, mode='train',
                 shuffle=True, img_size=[64, 64],label_size=[128, 128], buffer_size=1000,FLAGS=None):
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

        # retrieve the data from the dictionary
        self._load_data()

        # number of samples in the dataset
        self.data_size = len(self.label_paths)

        self.img_size = img_size  # (64, 64)
        self.label_size = label_size  # (128, 128)
        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor (build in the system)
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.label_paths = convert_to_tensor(self.label_paths, dtype=dtypes.string) # it is a path, too.

        output = tf.train.slice_input_producer([self.img_paths, self.label_paths],
                                               shuffle=False, capacity=FLAGS.name_queue_capacity)

        # Reading and decode the images
        reader = tf.WholeFileReader(name='image_reader')
        image_LR = tf.read_file(output[0])
        image_HR = tf.read_file(output[1])

        input_image_LR = tf.image.decode_png(image_LR, channels=3)#, channels=0)
        input_image_HR = tf.image.decode_png(image_HR, channels=3)#, channels=0)

        input_image_LR = tf.expand_dims(input_image_LR[:,:,0],-1)
        input_image_HR = tf.expand_dims(input_image_HR[:,:,0],-1)

        # input_image_LR = tf.cast(input_image_LR, tf.float32)
        # input_image_HR = tf.cast(input_image_HR, tf.float32)
        input_image_LR = tf.image.convert_image_dtype(input_image_LR, dtype=tf.float32)
        input_image_HR = tf.image.convert_image_dtype(input_image_HR, dtype=tf.float32)

        input_image_LR = tf.identity(input_image_LR)
        input_image_HR = tf.identity(input_image_HR)

        # Normalize the low resolution image to [0, 1], high resolution to [-1, 1]
        a_image = preprocessLR(input_image_LR)
        b_image = preprocess(input_image_HR)  # label:[-1,1]

        inputs, targets = [a_image, b_image] #[input_image_LR, input_image_HR]#

        with tf.name_scope('data_preprocessing'):
            with tf.variable_scope('random_flip'):
                # Check for random flip:
                if (FLAGS.flip is True) and (FLAGS.mode == 'train'):
                    print('[Config] Use random flip')
                    # Produce the decision of random flip
                    decision = tf.random_uniform([], 0, 1, dtype=tf.float32)

                    input_images = random_flip(inputs, decision)
                    target_images = random_flip(targets, decision)
                else:
                    input_images = tf.identity(inputs)
                    target_images = tf.identity(targets)

        self.input_images = input_images
        self.target_images = target_images

        '''
        #----------------------------------------------------------
        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.label_paths))

        # distinguish between train/infer. when calling the parsing functions
        #if mode == 'train':
        data = data.map(map_func=self._parse_function_train, num_parallel_calls=8)

        data = data.prefetch(buffer_size)
        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)


        # create a new dataset with batches of images
        #data = data.batch(batch_size)
        self.data = data
        '''
    def _load_data(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.label_paths = []

        path = self.file_path+self.train_data_dic
        print('path',path)
        for i in os.listdir(path):
            self.img_paths.append(self.file_path + self.train_data_dic + i)
            self.label_paths.append(self.file_path + self.train_label_dic + i)


    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.label_paths
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.label_paths = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.label_paths.append(labels[i])

    def _parse_function_train(self, image_path, label_path):
        """Input parser for samples of the training set."""

        # load and preprocess the image
        img_string = tf.read_file(image_path)
        inputs = tf.image.decode_jpeg(img_string, channels=0)
        # convert label to ground truth png
        img_string2 = tf.read_file(label_path)
        targets = tf.image.decode_png(img_string2)

        # The data augmentation part
        with tf.name_scope('data_preprocessing'):
            with tf.variable_scope('random_flip'):
                # Check for random flip:
                if (FLAGS.flip is True) and (FLAGS.mode == 'train'):
                    print('[Config] Use random flip')
                    # Produce the decision of random flip
                    decision = tf.random_uniform([], 0, 1, dtype=tf.float32)

                    input_images = random_flip(inputs, decision)
                    target_images = random_flip(targets, decision)
                else:
                    input_images = tf.identity(inputs)
                    target_images = tf.identity(targets)

        return img_decoded, img_decoded2 #img_bgr, img_gt


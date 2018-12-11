# -*- coding: utf-8 -*-
import os, sys

sys.path.append(os.getcwd())

import time
from glob import glob
from data_processing import *
import matplotlib
import cv2
from tensorflow.contrib.data import Iterator
matplotlib.use('Agg')
import numpy as np
import sklearn.datasets
import tensorflow as tf
from six.moves import xrange
import math

locale.setlocale(locale.LC_ALL, '')
flags = tf.app.flags
flags.DEFINE_string("train", "True", "train")
flags.DEFINE_string("train_file", '/Users/zpy/Desktop/xray_images/', "data path")
flags.DEFINE_integer("BATCH_SIZE", "128", "BATCH_SIZE")
flags.DEFINE_integer("DATA_HEIGHT", "64", "DATA_HEIGHT")
flags.DEFINE_integer("DATA_WIDTH", "64", "DATA_WIDTH")
flags.DEFINE_integer("LABEL_HEIGHT", "128", "LABEL_HEIGHT")
flags.DEFINE_integer("LABEL_WEIGHT", "128", "LABEL_WEIGHT")
flags.DEFINE_string("train_data_dic","train_images_64x64/","train_data_dic")
flags.DEFINE_string("train_label_dic", "train_images_128x128/", "train_label_dic")
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, sess, ITERS=30000, num_epochs=10,
                 checkpoint_dir=None, cost_dir=None, dataset_name=None):
        self.sess = sess
        self.BATCH_SIZE = FLAGS.BATCH_SIZE
        self.ITERS = ITERS
        self.checkpoint_dir = checkpoint_dir
        self.cost_dir = cost_dir
        self.num_epochs = num_epochs
        self.CHANNEL = 1
        self.dataset_name = dataset_name
        self.DATA_HEIGHT = FLAGS.DATA_HEIGHT
        self.DATA_WIDTH = FLAGS.DATA_WIDTH
        self.LABEL_HEIGHT = FLAGS.LABEL_HEIGHT
        self.LABEL_WIDTH = FLAGS.LABEL_WIDTH

        self.build()

    def build(self):
        self.global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # self.train_op=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).
            # minimize(self.cost, var_list=vgg_params)
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=1e-4,
                beta1=0.9,
                beta2=0.99
            ).minimize(self.cost, global_step=self.global_step)  # tf.train.GradientDescentOptimizer
        self.saver = tf.train.Saver()

    def LeakyReLU(self, x, alpha=0.2):
        return tf.maximum(alpha * x, x)

    def ReLU(self, output):
        return tf.nn.relu(output)

    def BN(self, name, axis, inputs, is_training=True):
        return (tf.layers.batch_normalization(inputs, axis=axis, momentum=0.1, epsilon=1e-5,
                                              training=is_training, name=name))

    def Conv2D(self, name, input_dim, output_dim, filter_size, inputs, stride=2,
               he_init=True, gain=1.):

        with tf.name_scope(name) as scope:
            def uniform(stdev, size):
                return np.random.uniform(
                    low=-stdev * np.sqrt(3),
                    high=stdev * np.sqrt(3),
                    size=size
                ).astype('float32')

            fan_in = input_dim * filter_size[0] * filter_size[1]  # filter_size**2
            fan_out = output_dim * filter_size[0] * filter_size[1] / (stride ** 2)  # filter_size**2

            if he_init:
                filters_stdev = np.sqrt(4. / (fan_in + fan_out))
            else:  # Normalized init (Glorot & Bengio)
                filters_stdev = np.sqrt(2. / (fan_in + fan_out))

            filter_values = uniform(
                filters_stdev,
                (filter_size[0], filter_size[1], input_dim, output_dim))

            filter_values *= gain
            filters = param(name + '.Filters', filter_values)

            result = tf.nn.conv2d(
                input=inputs,
                filter=filters,
                strides=[1, stride, stride, 1],
                padding='SAME',
                data_format='NHWC'
            )

            return result

    def ConvLayer(self, name, input, n_out, filter_size=[3, 3], stride=2):
        # name, input_dim, output_dim, filter_size, inputs
        output = self.Conv2D(name + '_1', input.get_shape().as_list()[-1], n_out, filter_size, input, stride)
        print(name, output)
        output = self.BN(name + "_BN", 0, output)
        output = self.ReLU(output)
        return output

    def bilinear_upsample_weights(self, input_dim, output_dim):
        """
        Create weights matrix for transposed convolution with bilinear filter nitialization.
        """
        # 4 times
        factor = 4
        filter_size = 2 * factor - factor % 2

        weights = np.zeros((filter_size,
                            filter_size,
                            output_dim,
                            input_dim), dtype=np.float32)

        upsample_kernel = self.upsample_filt(filter_size)
        for i in range(output_dim):
            for k in range(input_dim):
                weights[:, :, i, k] = upsample_kernel

        return weights

    def Deconv2D(self, name, input_dim, output_shape, filter_size, inputs, stride):
        with tf.name_scope(name) as scope:
            output_dim = output_shape[-1]
            # number_of_classes=5
            filter_values = self.bilinear_upsample_weights(input_dim, output_dim)
            print(name, filter_values.shape)

            filters = param(
                name + '.Filters',
                filter_values)

            result = tf.nn.conv2d_transpose(
                value=inputs,
                filter=filters,
                output_shape=output_shape,
                strides=[1, stride, stride, 1],
                padding='SAME')
        return result

    def train(self):
        tf.global_variables_initializer().run()

        # load ckpt
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        tr_data = ImageDataGenerator(FLAGS.train_file,
                                     img_size=[self.DATA_HEIGHT, self.DATA_WIDTH],
                                     label_size=[self.LABEL_HEIGHT, self.LABEL_WEIGHT],
                                     batch_size=self.BATCH_SIZE,
                                     shuffle=True)

        print('[Ves9] Data done ...')
        # create an reinitializable iterator given the dataset structure
        iterator = Iterator.from_structure(tr_data.data.output_types,
                                           tr_data.data.output_shapes)
        next_batch = iterator.get_next()
        batch_idxs = int(np.floor(tr_data.data_size / self.BATCH_SIZE))
        print(tr_data.data_size)

        for epoch in range(self.num_epochs):  #
            print('[Ves9] Begin epoch ', epoch, '...')
            for iteration in range(batch_idxs):

                start_time = time.time()
                batch_images, batch_gts = self.sess.run(next_batch)

                _cost, result, acc_, _ = self.sess.run(
                    [self.cost, self.output, self.accuracy, self.train_op],
                    feed_dict={self.images: batch_images,
                               self.ground_truth: batch_gts})
                tm = time.time() - start_time

                if np.mod(iteration, 5000) == 0 and iteration != 0:
                    print('[INFO] Save checkpoint...')
                    self.save(self.checkpoint_dir, iteration)
                if (iteration % 50 == 0):
                    print(iteration, ': ', '%.2f' % (tm), _cost)

            print('[INFO] Save the epoch checkpoint ... ')
            self.save(self.checkpoint_dir)
        # Last saving
        print('[INFO] Save the last checkpoint...')
        self.save(self.checkpoint_dir)

    @property
    def model_dir(self):
        return "{}_{}".format(
            self.dataset_name, self.BATCH_SIZE)

    def save(self, checkpoint_dir, step=0):
        model_name = "Vgg9_res.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=self.global_step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def load_ckpt(self, ckpt):
        print(" [*] Reading checkpoints...")
        self.saver.restore(self.sess, ckpt)


def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = False

    with tf.Session(config=run_config) as sess:
        model = Model(sess=sess,
                      checkpoint_dir=checkpoint_dir,
                      cost_dir=cost_dir)

        if(FLAGS.train =='True'):
            model.train()

        elif(FLAGS.train =='False'): # test
            print('Test.....')
            #ves.test_pipeline()
            #ves.test()



if __name__ == '__main__':
    tf.app.run()
# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.data import Iterator
from lib.tool import generator, SRGAN, save_images
from lib.ops import *
import math
import time
sys.path.append(os.getcwd())
from glob import glob
from data_processing import *
import matplotlib
import cv2
matplotlib.use('Agg')


flags = tf.app.flags
flags.DEFINE_string("train", "True", "train")##duoyu

flags.DEFINE_integer("DATA_HEIGHT", "64", "DATA_HEIGHT")
flags.DEFINE_integer("DATA_WIDTH", "64", "DATA_WIDTH")
flags.DEFINE_integer("LABEL_HEIGHT", "128", "LABEL_HEIGHT")
flags.DEFINE_integer("LABEL_WIDTH", "128", "LABEL_WIDTH")
flags.DEFINE_string("gpu","2","gpu")
flags.DEFINE_string("train_data_dic","train_images_64x64/","train_data_dic")
flags.DEFINE_string("train_label_dic", "train_images_128x128/", "train_label_dic")

flags.DEFINE_string("inference_dir", "./inferences/", "inference_dir")


# The system parameter

flags.DEFINE_string("checkpoint", "/home/nfs/zpy/xrays/HighResImg_Gen/checkpoints2/model-70000", "checkpoint")
flags.DEFINE_string("checkpoint_dir", "./checkpoints3/", "checkpoint_dir")
flags.DEFINE_string("generation_dir", "./generations3/", "generations_dir")
flags.DEFINE_string('summary_dir', "./summarys3/", 'The dirctory to output the summary')
flags.DEFINE_string('mode', 'train', 'The mode of the model train, test.')
flags.DEFINE_boolean('pre_trained_model', False, 'pretrain')
flags.DEFINE_string('pre_trained_model_type', 'SRGAN', 'The type of pretrained model (SRGAN or SRResnet)')
flags.DEFINE_boolean('is_training', True, 'Training => True, Testing => False')
flags.DEFINE_string('vgg_ckpt', './vgg19/vgg_19.ckpt', 'path to checkpoint file for the vgg19')
flags.DEFINE_string('task', 'SRGAN', 'The task: SRGAN, SRResnet')

# The data preparing operation
flags.DEFINE_integer("batch_size", "32", "BATCH_SIZE")
flags.DEFINE_boolean('flip', True, 'Whether random flip data augmentation is applied')
flags.DEFINE_string("train_file", '/home/nfs/zpy/xray_images/', "data path")

flags.DEFINE_integer('name_queue_capacity', 2048, 'The capacity of the filename queue (suggest large to ensure enough random shuffle.')
flags.DEFINE_integer('image_queue_capacity', 2048, 'The capacity of the image queue (suggest large to ensure enough random shuffle')
flags.DEFINE_integer('queue_thread', 10, 'The threads of the queue (More threads can speedup the training process.')

# Generator configuration
flags.DEFINE_integer('num_resblock', 8, 'How many residual blocks are there in the generator')
# The content loss parameter
flags.DEFINE_string('perceptual_mode', 'MSE', 'The type of feature used in perceptual loss')
flags.DEFINE_float('EPS', 1e-12, 'The eps added to prevent nan')
flags.DEFINE_float('ratio', 0.001, 'The ratio between content loss and adversarial loss')
flags.DEFINE_float('vgg_scaling', 0.0061, 'The scaling factor for the perceptual loss if using vgg perceptual loss')
# The training parameters
flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
flags.DEFINE_integer('summary_freq', 30, 'The frequency of writing summary')
flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving images')

FLAGS = flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
print_configuration_op(FLAGS)


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

        '''
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
        '''



    def train(self):
        tf.global_variables_initializer().run()
        could_load=0
        # load ckpt
        #could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        tr_data = ImageDataGenerator(FLAGS.train_file,
                                     img_size=[self.DATA_HEIGHT, self.DATA_WIDTH],
                                     label_size=[self.LABEL_HEIGHT, self.LABEL_WIDTH],
                                     batch_size=self.BATCH_SIZE,
                                     shuffle=True)

        print('[Model] Data done ...')
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
                print(batch_images[0].shape, batch_images[0])
                print(batch_gts[0].shape, batch_gts[0])
                '''
                _cost, result, acc_, _ = self.sess.run(
                    [self.cost, self.output, self.accuracy, self.train_op],
                    feed_dict={self.images: batch_images,
                               self.ground_truth: batch_gts})
                '''
                tm = time.time() - start_time
                '''
                if np.mod(iteration, 5000) == 0 and iteration != 0:
                    print('[INFO] Save checkpoint...')
                    self.save(self.checkpoint_dir, iteration)
                if (iteration % 50 == 0):
                    print(iteration, ': ', '%.2f' % (tm), _cost)
                '''
            print('[INFO] Save the epoch checkpoint ... ')
            #self.save(self.checkpoint_dir)
        # Last saving
        # print('[INFO] Save the last checkpoint...')
        # self.save(self.checkpoint_dir)

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
    # The testing mode
    if FLAGS.mode == 'test':
        # Check the checkpoint
        if FLAGS.checkpoint is None:
            raise ValueError('The checkpoint file is needed to performing the test.')

        # In the testing time, no flip and crop is needed
        if FLAGS.flip == True:
            FLAGS.flip = False

        # if FLAGS.crop_size is not None:
        #    FLAGS.crop_size = None

        # Declare the test data reader
        test_data = test_data_loader(FLAGS)

        inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='inputs_raw')
        targets_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='targets_raw')
        path_LR = tf.placeholder(tf.string, shape=[], name='path_LR')
        path_HR = tf.placeholder(tf.string, shape=[], name='path_HR')

        with tf.variable_scope('generator'):
            if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
                gen_output = generator(inputs_raw, 3, reuse=False, FLAGS=FLAGS)
            else:
                raise NotImplementedError('Unknown task!!')

        print('Finish building the network')

        with tf.name_scope('convert_image'):
            # Deprocess the images outputed from the model
            inputs = deprocessLR(inputs_raw)
            targets = deprocess(targets_raw)
            outputs = deprocess(gen_output)

            # Convert back to uint8
            converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
            converted_targets = tf.image.convert_image_dtype(targets, dtype=tf.uint8, saturate=True)
            converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

        with tf.name_scope('encode_image'):
            save_fetch = {
                "path_LR": path_LR,
                "path_HR": path_HR,
                "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name='input_pngs'),
                "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs'),
                "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name='target_pngs')
            }

        # Define the weight initiallizer (In inference time, we only need to restore the weight of the generator)
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        weight_initiallizer = tf.train.Saver(var_list)

        # Define the initialization operation
        init_op = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Load the pretrained model
            print('Loading weights from the pre-trained model')
            weight_initiallizer.restore(sess, FLAGS.checkpoint)

            max_iter = len(test_data.inputs)
            print('Evaluation starts!!')
            for i in range(max_iter):
                input_im = np.array([test_data.inputs[i]]).astype(np.float32)
                target_im = np.array([test_data.targets[i]]).astype(np.float32)
                path_lr = test_data.paths_LR[i]
                path_hr = test_data.paths_HR[i]
                results = sess.run(save_fetch, feed_dict={inputs_raw: input_im, targets_raw: target_im,
                                                          path_LR: path_lr, path_HR: path_hr})
                filesets = save_images(results, FLAGS)
                for i, f in enumerate(filesets):
                    print('evaluate image', f['name'])

    # the inference mode (just perform super resolution on the input image)
    elif FLAGS.mode == 'inference':
        # Check the checkpoint
        if FLAGS.checkpoint is None:
            raise ValueError('The checkpoint file is needed to performing the test.')

        # In the testing time, no flip and crop is needed
        if FLAGS.flip == True:
            FLAGS.flip = False

        #if FLAGS.crop_size is not None:
        #    FLAGS.crop_size = None

        # Declare the test data reader
        inference_data = inference_data_loader(FLAGS)

        inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 1], name='inputs_raw')
        path_LR = tf.placeholder(tf.string, shape=[], name='path_LR')

        with tf.variable_scope('generator'):
            if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
                gen_output = generator(inputs_raw, 1, reuse=False, FLAGS=FLAGS)
            else:
                raise NotImplementedError('Unknown task!!')
        print('Finish building the network')

        with tf.name_scope('convert_image'):
            # Convert back to uint8
            converted_inputs = tf.image.convert_image_dtype(inputs_raw, dtype=tf.uint8, saturate=True)
            converted_outputs = tf.image.convert_image_dtype(gen_output, dtype=tf.uint8, saturate=True)

        with tf.name_scope('encode_image'):
            save_fetch = {
                "path_LR": path_LR,
                "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name='input_pngs'),
                "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs')
            }

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        weight_initiallizer = tf.train.Saver(var_list)
        init_op = tf.global_variables_initializer()

        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True

        with tf.Session(config=run_config) as sess:
            # Load the pretrained model
            print('Loading weights from the pre-trained model')
            weight_initiallizer.restore(sess, FLAGS.checkpoint)
            max_iter = inference_data.size
            print('Evaluation starts!!')
            for i in range(max_iter):
                input_im = np.array([inference_data.inputs[i]]).astype(np.float32)
                path_lr = inference_data.paths_LR[i]
                gen_output, results = sess.run([converted_outputs, save_fetch], feed_dict={inputs_raw: input_im, path_LR: path_lr})
                # print('img', gen_output.shape, gen_output[0,0,0], type(gen_output[0,0,0]))

                filesets = save_images(results, gen_output, FLAGS)
                #for i, f in enumerate(filesets):
                #    print('evaluate image', f['name'])

    elif (FLAGS.mode == 'train'):
        data = data_loader(FLAGS)
        print('Data count = %d' % (data.image_count))

        # Connect to the network
        if FLAGS.task == 'SRGAN':
            Net = SRGAN(data.inputs, data.targets, FLAGS)
        else:
            raise NotImplementedError('Unknown task type')

        print('Finish building the network!!!')

        # Define the saver and weight initiallizer
        saver = tf.train.Saver(max_to_keep=10)
        # The variable list
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        if FLAGS.task == 'SRGAN':
            tf.summary.scalar('discriminator_loss', Net.discrim_loss)
            tf.summary.scalar('adversarial_loss', Net.adversarial_loss)
            tf.summary.scalar('content_loss', Net.content_loss)
            tf.summary.scalar('generator_loss', Net.content_loss + FLAGS.ratio * Net.adversarial_loss)
            # tf.summary.scalar('PSNR', psnr)
            tf.summary.scalar('learning_rate', Net.learning_rate)

            if FLAGS.pre_trained_model_type == 'SRGAN':
                var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator') + \
                            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        if not FLAGS.perceptual_mode == 'MSE':
            vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
            vgg_restore = tf.train.Saver(vgg_var_list)

        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        # Use superviser to coordinate all queue and summary writer
        sv = tf.train.Supervisor(logdir=FLAGS.summary_dir, save_summaries_secs=0, saver=None)
        with sv.managed_session(config=run_config) as sess:
            '''
            model = Model(sess=sess,
                          checkpoint_dir=None,
                          cost_dir=None)

            # if(FLAGS.train =='True'):
            # model.train()
            '''

            if (FLAGS.checkpoint is not None) and (FLAGS.pre_trained_model is False):
                print('Loading model from the checkpoint...')
                checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
                saver.restore(sess, checkpoint)

            elif (FLAGS.checkpoint is not None) and (FLAGS.pre_trained_model is True):
                print('Loading weights from the pre-trained model')
                weight_initiallizer.restore(sess, FLAGS.checkpoint)

            if FLAGS.max_epoch is None:
                if FLAGS.max_iter is None:
                    raise ValueError('one of max_epoch or max_iter should be provided')
                else:
                    max_iter = FLAGS.max_iter
            else:
                max_iter = FLAGS.max_epoch * data.steps_per_epoch

            # create an reinitializable iterator given the dataset structure
            # iterator = Iterator.from_structure(tr_data.data.output_types,
            #                                    tr_data.data.output_shapes)
            # next_batch = iterator.get_next()
            # batch_idxs = int(np.floor(tr_data.data_size / FLAGS.batch_size))
            # max_iter = FLAGS.max_epoch * batch_idxs
            # print(tr_data.data_size)
            print('Optimization starts!!!')
            start = time.time()
            for step in range(max_iter):
                fetches = {
                    "train": Net.train,
                    "global_step": sv.global_step,
                }

                # batch_images, batch_gts = self.sess.run(next_batch)
                # print(batch_images[0].shape, batch_images[0])
                # print(batch_gts[0].shape, batch_gts[0])

                if ((step + 1) % FLAGS.display_freq) == 0:
                    if FLAGS.task == 'SRGAN':
                        fetches["discrim_loss"] = Net.discrim_loss
                        fetches["adversarial_loss"] = Net.adversarial_loss
                        fetches["content_loss"] = Net.content_loss
                        #fetches["PSNR"] = psnr
                        fetches["learning_rate"] = Net.learning_rate
                        fetches["global_step"] = Net.global_step

                if ((step + 1) % FLAGS.summary_freq) == 0:
                    fetches["summary"] = sv.summary_op
                results = sess.run(fetches)
                if ((step + 1) % FLAGS.summary_freq) == 0:
                    print('Recording summary!!')
                    sv.summary_writer.add_summary(results['summary'], results['global_step'])

                if ((step + 1) % FLAGS.display_freq) == 0:
                    train_epoch = math.ceil(results["global_step"] / data.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % data.steps_per_epoch + 1
                    rate = (step + 1) * FLAGS.batch_size / (time.time() - start)
                    remaining = (max_iter - step) * FLAGS.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                    train_epoch, train_step, rate, remaining / 60))
                    if FLAGS.task == 'SRGAN':
                        print("global_step", results["global_step"])
                        #print("PSNR", results["PSNR"])
                        print("discrim_loss", results["discrim_loss"])
                        print("adversarial_loss", results["adversarial_loss"])
                        print("content_loss", results["content_loss"])
                        print("learning_rate", results['learning_rate'])

                if ((step + 1) % FLAGS.save_freq) == 0:
                    print('Save the checkpoint')
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model'), global_step=sv.global_step)

            print('Optimization done!!!!!!!!!!!!')


if __name__ == '__main__':
    tf.app.run()
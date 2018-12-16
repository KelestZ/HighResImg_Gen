import tensorflow as tf
import tensorflow.contrib.slim as slim


'''
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

	# Define our Lrelu
def ConvLayer(self, name, input, n_out, filter_size=[3, 3], stride=2):
	# name, input_dim, output_dim, filter_size, inputs
	output = self.Conv2D(name + '_1', input.get_shape().as_list()[-1], n_out, filter_size, input, stride)
	print(name, output)
	output = self.BN(name + "_BN", 0, output)
	output = self.ReLU(output)
	return output

'''

def lrelu(inputs, alpha):
	return tf.keras.layers.LeakyReLU(alpha=alpha).call(inputs)

# Define our tensorflow version PRelu
# Define our tensorflow version PRelu
def prelu_tf(inputs, name='Prelu'):
	with tf.variable_scope(name):
		alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
	pos = tf.nn.relu(inputs)
	neg = alphas * (inputs - abs(inputs)) * 0.5
	return pos + neg


def batchnorm(inputs, is_training):
	return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
							scale=False, fused=True, is_training=is_training)

def denselayer(inputs, output_size):
	output = tf.layers.dense(inputs, output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
	return output

# Define the convolution building block
def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
	# kernel: An integer specifying the width and height of the 2D convolution window
	with tf.variable_scope(scope):
		if use_bias:
			return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
							   activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
		else:
			return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
							   activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
							   biases_initializer=None)


# The operation used to print out the configuration
def print_configuration_op(FLAGS):
	print('[Configurations]:')
	a = FLAGS.mode
	#pdb.set_trace()
	for name, value in FLAGS.__flags.items():
		if type(value) == float:
			print('\t%s: %f'%(name, value))
		elif type(value) == int:
			print('\t%s: %d'%(name, value))
		elif type(value) == str:
			print('\t%s: %s'%(name, value))
		elif type(value) == bool:
			print('\t%s: %s'%(name, value))
		else:
			print('\t%s: %s' % (name, value))
	print('End of configuration')


def random_flip(input, decision):
	f1 = tf.identity(input)
	f2 = tf.image.flip_left_right(input)
	output = tf.cond(tf.less(decision, 0.5), lambda: f2, lambda: f1)
	return output

# The implementation of PixelShuffler
def pixelShuffler(inputs, scale=2):
	size = tf.shape(inputs)
	batch_size = size[0]
	h = size[1]
	w = size[2]
	c = inputs.get_shape().as_list()[-1]

	# Get the target channel size
	channel_target = c // (scale * scale)
	channel_factor = c // channel_target

	shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
	shape_2 = [batch_size, h * scale, w * scale, 1]

	# Reshape and transpose for periodic shuffling for each channel
	input_split = tf.split(inputs, channel_target, axis=3)
	output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

	return output
def phaseShift(inputs, scale, shape_1, shape_2):
	# Tackle the condition when the batch is None
	X = tf.reshape(inputs, shape_1)
	X = tf.transpose(X, [0, 1, 3, 2, 4])

	return tf.reshape(X, shape_2)

def compute_psnr(ref, target):
	ref = tf.cast(ref, tf.float32)
	target = tf.cast(target, tf.float32)
	diff = target - ref
	sqr = tf.multiply(diff, diff)
	err = tf.reduce_sum(sqr)
	v = tf.shape(diff)[0] * tf.shape(diff)[1] * tf.shape(diff)[2] * tf.shape(diff)[3]
	mse = err / tf.cast(v, tf.float32)
	psnr = 10. * (tf.log(255. * 255. / mse) / tf.log(10.))

	return psnr

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


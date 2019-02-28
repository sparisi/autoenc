'''
* This implementation is supposed to be used with normalized images (the input are pixels whose value is divided by 255).
* For this reason, the output activation is a sigmoid.
* All hidden layers have a relu activation (feel free to change it).
* AE is a simple (deep) AE whose input is a 1D vector (flattened images).
* CAE2 applies 2D convolution to multidimensional vectors (the input can be a gray image, an RGB image, or a series of images).
* For convolutional AEe, the input has always to be reshaped to have 3 dims.
* All AEe return 1) the low-dimensional (encoded) image given the original one, 2) return the reconstructed (decoded) image given the encoded one.
'''

import tensorflow as tf

class AE:
	def __init__(self, x, sizes, scope):
		sizes = [x.get_shape().as_list()[-1]] + sizes
		with tf.variable_scope(scope):
			last_out = x
			for l, size in enumerate(sizes[1:]): # build encode layers
				last_out = tf.layers.dense(last_out, size, activation=tf.nn.relu, name='encode'+str(l))
			self.encode = last_out
			for l, size in enumerate(reversed(sizes[:-1])): # build decode layers
				activation = tf.nn.relu
				if l == len(sizes[:-1]) - 1:
					activation = tf.nn.sigmoid
				last_out = tf.layers.dense(last_out, size, activation=activation, name='decode'+str(l))
			self.decode = last_out
		self.vars = tf.trainable_variables(scope=scope)

class CAE2:
	def __init__(self, x, sizes, scope):
		# sizes is a list of tuples (channel size, filter size 1, filter size 2)
		# eg (128, 3, 3) corresponds to a layer with 128 filters of size 3x3
		with tf.variable_scope(scope):
			last_out = x
			for l, (filters, rows, cols) in enumerate(sizes): # build encode layers
				last_out = tf.layers.conv2d(last_out, filters, (rows, cols), padding='same', activation=tf.nn.relu, name='encode_conv'+str(l))
				last_out = tf.layers.max_pooling2d(last_out, pool_size=(2,2), strides=(2,2), padding='same', name='encode_maxpool'+str(l))
			self.encode = last_out
			for l, (filters, rows, cols) in enumerate(reversed(sizes)): # build decode layers
				last_out = tf.layers.conv2d(last_out, filters, (rows, cols), padding='same', activation=tf.nn.relu, name='decode_conv'+str(l))
				last_out = tf.keras.layers.UpSampling2D(name='decode_upsamp'+str(l))(last_out) # tf.layers.up_sampling2d does not exist
			last_out = tf.layers.conv2d(last_out, x.get_shape().as_list()[-1], (sizes[0][1], sizes[0][2]), padding='same', activation=tf.nn.sigmoid, name='decode_conv'+str(l+1))
			self.decode = last_out
		self.vars = tf.trainable_variables(scope=scope)

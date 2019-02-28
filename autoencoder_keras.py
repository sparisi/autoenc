'''
A collection of some autoencoders presented in
http://blog.keras.io/building-autoencoders-in-keras.html

* This implementation is supposed to be used with normalized images (the input are pixels whose value is divided by 255).
* For this reason, the output activation is a sigmoid.
* All hidden layers have a relu activation (feel free to change it).
* Autoencoder is a simple (deep) AE whose input is a 1D vector (flattened images).
* ConvolutionalAutoencoder1D applies 1D convolution to 1D vector.
* ConvolutionalAutoencoder2D applies 2D convolution to multidimensional vectors (the input can be a gray image, an RGB image, or a series of images).
* For convolutional AEe, the input has always to be reshaped to have 3 dims.
* Beside the default "predict" function, all return AEe 1) the low-dimensional (encoded) image given the original one, 2) the reconstructed (decoded) image given the encoded one.
'''

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, MaxPooling1D, UpSampling1D, Conv3D, MaxPooling3D, UpSampling3D
from keras.callbacks import EarlyStopping


class AutoencoderBase(object):
	def train(self, x_train, x_test, epochs, batch_size, stop_early=True):
		callbacks = []
		if stop_early:
			callbacks.append(EarlyStopping(monitor='val_loss',
                                patience=10,
                                verbose=1,
                                mode='auto',
                                restore_best_weights=True))

		self.autoencoder.fit(x_train, x_train,
				epochs=epochs,
				batch_size=batch_size,
				shuffle=True,
				validation_data=(x_test, x_test),
				callbacks=callbacks)

	def predict(self, x):
		return self.autoencoder.predict(x)

	def encode(self, x):
		return self.encoder.predict(x)

	def decode(self, x):
		return self.decoder.predict(x)

	def summary(self):
		self.autoencoder.summary()


class Autoencoder(AutoencoderBase):
	def __init__(self, dims):
		dim_in = dims[0] # original image size
		dim_out = dims[-1] # encoded image size
		dims_encoder = dims[1:] # hidden layers size
		dims_decoding = dims[:-1]
		dims_decoding.reverse()

		input_img = Input(shape=(dim_in,), name='EncoderIn')
		decoder_input = Input(shape=(dim_out,), name='DecoderIn')

		# construct encoder layers
		encoded = input_img
		for i, dim in enumerate(dims_encoder):
			encoded = Dense(dim, activation='relu', name='Encoder{0}'.format(i))(encoded)

		# construct decoder layers (decoded is connected to encoded, whereas encoded is not connected to decoded)
		decoded = encoded
		decoder = decoder_input
		for i, dim in enumerate(dims_decoding):
			activation = 'relu'
			if i == len(dims_decoding) - 1:
				activation = 'sigmoid'
			layer = Dense(dim, activation=activation, name='Decoder{0}'.format(i))
			decoded = layer(decoded)
			decoder = layer(decoder)

		self.autoencoder = Model(input_img, decoded)
		self.encoder = Model(input_img, encoded)
		self.decoder = Model(decoder_input, decoder)
		self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


class ConvolutionalAutoencoder1D(AutoencoderBase):
	def __init__(self, dims, dim_in):
		# dims is a list of tuples (channel size, filter size)
		# eg (128, 9) corresponds to a layer with 128 filters of size 9
		input_img = Input(shape=(dim_in, 1), name='EncoderIn')

        # each layer halves the size of the image
		decoder_input = Input(shape=(int(dim_in/2**len(dims)), dims[-1][0]), name='DecoderIn')

		# construct encoder layers
		encoded = input_img
		for i, (filters, size) in enumerate(dims):
			encoded = Conv1D(filters, size, activation='relu', padding='same', name='Conv{0}'.format(i))(encoded)
			encoded = MaxPooling1D(padding='same', name='MaxPool{0}'.format(i))(encoded)

		# construct decoder layers (decoded is connected to encoded, whereas encoded is not connected to decoded)
		decoded = encoded
		decoder = decoder_input
		for i, (filters, size) in enumerate(reversed(dims)):
			layer = Conv1D(filters, size, activation='relu', padding='same', name='Deconv{0}'.format(i))
			decoded = layer(decoded)
			decoder = layer(decoder)
			layer = UpSampling1D(name='UpSampling{0}'.format(i))
			decoded = layer(decoded)
			decoder = layer(decoder)

		# reduce from X filters to 1 in the output layer
		layer = Conv1D(1, dims[0][1], activation='sigmoid', padding='same', name='DecoderOut')
		decoded = layer(decoded)
		decoder = layer(decoder)

		self.autoencoder = Model(input_img, decoded)
		self.encoder = Model(input_img, encoded)
		self.decoder = Model(decoder_input, decoder)
		self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


class ConvolutionalAutoencoder2D(AutoencoderBase):
	def __init__(self, dims, h, w, c):
		# dims is a list of tuples (channel size, filter size 1, filter size 2)
		# eg (128, 3, 3) corresponds to a layer with 128 filters of size 3x3

        # c is the number of channels (3 for rgb images, more if the input is a sequence of images)
		input_img = Input(shape=(h, w, c), name='EncoderIn')

        # each layer halves the size of the image
		decoder_input = Input(shape=(int(h/2**len(dims)), int(w/2**len(dims)), dims[-1][0]), name='DecoderIn')

		# construct encoder layers
		encoded = input_img
		for i, (filters, rows, cols) in enumerate(dims):
			encoded = Conv2D(filters, (rows, cols), activation='relu', padding='same', name='Conv{0}'.format(i))(encoded)
			encoded = MaxPooling2D(padding='same', name='MaxPool{0}'.format(i))(encoded)

		# construct decoder layers (decoded is connected to encoded, whereas encoded is not connected to decoded)
		decoded = encoded
		decoder = decoder_input
		for i, (filters, rows, cols) in enumerate(reversed(dims)):
			layer = Conv2D(filters, (rows, cols), activation='relu', padding='same', name='Deconv{0}'.format(i))
			decoded = layer(decoded)
			decoder = layer(decoder)
			layer = UpSampling2D(name='UpSampling{0}'.format(i))
			decoded = layer(decoded)
			decoder = layer(decoder)

		# reduce from X filters to 1 in the output layer
		layer = Conv2D(c, (dims[0][1], dims[0][2]), activation='sigmoid', padding='same', name='DecoderOut')
		decoded = layer(decoded)
		decoder = layer(decoder)

		self.autoencoder = Model(input_img, decoded)
		self.encoder = Model(input_img, encoded)
		self.decoder = Model(decoder_input, decoder)
		self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt

from autoencoder_keras import *

if __name__ == '__main__':

    mode = 'gray'
    mode = 'rgb'

    data = sio.loadmat('data.mat')
    h = np.asscalar(data['h2']) # height
    w = np.asscalar(data['w2']) # width
    X = data['I2'] if mode == 'rgb' else data['IG2']
    c = 3 if mode == 'rgb' else 1
    X = X / 255. # normalize data
    n_samples, data_dim = X.shape
    train_percentage = 0.90 # remainder is used for testing

    (x_train, x_test) = np.split(X, [int(n_samples * train_percentage)])
    x_train = (x_train.astype('float32'))
    x_test = (x_test.astype('float32'))

    # learning hyperparameters
    autoenc_type = 'cae2'
    epochs = 100
    batch_size = 32

    # create autoencoder
    if autoenc_type == 'ae':
        autoenc = Autoencoder([data_dim, 16, 32])
    elif autoenc_type == 'cae1':
        x_train = np.atleast_3d(x_train)
        x_test = np.atleast_3d(x_test)
        autoenc = ConvolutionalAutoencoder1D([(16, 2), (8, 1)], h*w)
    elif autoenc_type == 'cae2':
        x_train = np.reshape(x_train, (len(x_train), h, w, c), order="F")
        x_test = np.reshape(x_test, (len(x_test), h, w, c), order="F")
        autoenc = ConvolutionalAutoencoder2D([(16, 4, 4), (8, 4, 4)], h, w, c)
    else:
        raise Exception('Unknown autoencoder type: ' + str(autoenc_type))

    # train
    autoenc.summary()
    autoenc.train(x_train, x_test, epochs, batch_size)

    # test
    # decoded_imgs = autoenc.predict(x_test)
    encoded_imgs = autoenc.encode(x_test)
    decoded_imgs = autoenc.decode(encoded_imgs)

    # display original and decoded images
    n_display = 5
    fig = plt.figure()
    im_dims = np.array([h, w]) if mode == 'gray' else np.array([h, w, 3])
    for i in range(1, n_display+1):
        ax = plt.subplot(2, n_display, i)
        plt.imshow(x_test[i].reshape(im_dims, order="F")) # show original
        if mode == 'gray':
            plt.gray()
        plt.axis('off')

        ax = plt.subplot(2, n_display, i + n_display)
        plt.imshow(decoded_imgs[i].reshape(im_dims, order="F")) # show decoded
        if mode == 'gray':
            plt.gray()
        plt.axis('off')

    plt.show()

    # display final convolution
    if autoenc_type == 'cae2':
        fig = plt.figure()
        n_channels = 8
        for i in range(0, n_channels):
            ax = plt.subplot(1, n_channels, i+1)
            plt.imshow(encoded_imgs[0][:,:,i]) # show channel
            if mode == 'gray':
                plt.gray()
            plt.axis('off')

        plt.show()

    input("Press Enter to continue...")

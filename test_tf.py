import tensorflow as tf
import numpy as np

import scipy.io as sio

import matplotlib.pyplot as plt

from autoencoder_tf import *


def minibatch_idx_list(batch_size, dataset_size):
    batch_idx_list = np.random.choice(dataset_size, dataset_size, replace=False)
    for batch_start in range(0, dataset_size, batch_size):
        yield batch_idx_list[batch_start:min(batch_start + batch_size, dataset_size)]


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

    train_percentage = 0.9
    batch_size = 32
    epochs = 50
    (x_train, x_test) = np.split(X, [int(n_samples * train_percentage)])
    x_train = (x_train.astype('float32'))
    x_test = (x_test.astype('float32'))

    autoenc_type = 'cae2'

    if autoenc_type == 'ae':
        x_ph = tf.placeholder(dtype=tf.float32, shape=[None, data_dim], name='x')
        ae = AE(x_ph, [16, 32], 'ae')
    elif autoenc_type == 'cae2':
        x_ph = tf.placeholder(dtype=tf.float32, shape=[None, h, w, c], name='x')
        x_train = np.reshape(x_train, (len(x_train), h, w, c), order="F")
        x_test = np.reshape(x_test, (len(x_test), h, w, c), order="F")
        ae = CAE2(x_ph, [(16, 4, 4), (8, 4, 4)], 'ae')
    else:
        raise Exception('Unknown autoencoder type: ' + str(autoenc_type))

    loss_ae = tf.losses.mean_squared_error(ae.decode, x_ph)
    optimize_ae = tf.train.AdamOptimizer(1e-4).minimize(loss_ae, var_list=ae.vars)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # train
    for epoch in range(epochs):
        print('%d | loss_train: %e     loss_test: %e' % (epoch, session.run(loss_ae, {x_ph: x_train}), session.run(loss_ae, {x_ph: x_test})), flush=True)
        for batch_idx in minibatch_idx_list(batch_size, x_train.shape[0]):
            session.run(optimize_ae, {x_ph: x_train[batch_idx]})

    decoded_imgs = session.run(ae.decode, {x_ph: x_test})

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
    encoded_imgs = session.run(ae.encode, {x_ph: x_test})
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

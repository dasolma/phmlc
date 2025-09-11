from typing import Tuple

import numpy as np
import tensorflow as tf
from phm_framework import typing
from phm_framework.trainers.net import NetTrainer

EXTRA_CHANNEL = True
TRAINER = NetTrainer

def create_model(input_shape, output,
                 block_size: int = 2,
                 nblocks: int = 2,
                 kernel_size: typing.KernelSize = (1, 10),
                 l1: float = 1e-5, l2: float = 1e-4,
                 msblocks: int = 2,
                 dropout: float = 0.5,
                 filters: int = 64,
                 fc1: int = 256,
                 fc2: int = 128,
                 conv_activation: typing.Activation = 'relu',
                 dense_activation: typing.Activation = 'relu',
                 dilation_rate: int = 1,
                 batch_normalization: bool = True,
                 output_dim: int = 1):


    block_size = int(round(block_size))
    nblocks = int(round(nblocks))
    fc1 = int(round(fc1))
    fc2 = int(round(fc2))
    dilation_rate = int(round(dilation_rate))

    fc1 = int(fc1)
    fc2 = int(fc2)

    f1 = int(f1)
    f2 = int(f2)
    f3 = int(f3)
    ms_kernel_size = [f1, f2, f3]

    input_tensor = tf.keras.layers.Input(input_shape)
    x = input_tensor

    if input_shape[-3] > 5:
        x = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 3, 1]))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(5, kernel_size=(1,)))(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 3, 1, 2]))(x)

    for i, _ in enumerate(range(msblocks)):

        cblock = []
        for k in range(3):
            output_shape = x.shape
            f = ms_kernel_size[k]

            b = tf.keras.layers.Conv1D(filters, kernel_size=(f, 1), padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                                       kernel_initializer='he_uniform',
                                       name='MSConv_%d%d_%d' % (i, k, f),
                                       dilation_rate=dilation_rate)(x)

            if batch_normalization:
                b = tf.keras.layers.BatchNormalization()(b)
            b = tf.keras.layers.Activation(conv_activation)(b)

            cblock.append(b)

        x = tf.keras.layers.Add()(cblock)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)

    for i, n_cnn in enumerate([block_size] * nblocks):
        for j in range(n_cnn):
            x = tf.keras.layers.Conv1D(filters * 2 ** min(i, 2), kernel_size=kernel_size, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                                       kernel_initializer='he_uniform',
                                       dilation_rate=dilation_rate)(x)
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(conv_activation)(x)
        x = tf.keras.layers.MaxPooling2D((1, 2))(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)

    if np.prod(tf.shape(x)._inferred_value[1:]) * fc1 > 4e6:
        kernels = 2**(int(4e6 / (np.prod(tf.shape(x)._inferred_value[1:-1]) * fc1)).bit_length() - 1)

        x = tf.keras.layers.Conv2D(kernels, (1, 1))(x)
        x = tf.keras.layers.Activation(conv_activation)(x)

    x = tf.keras.layers.Flatten()(x)

    # FNN
    x = tf.keras.layers.Dense(fc1,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = tf.keras.layers.Activation(dense_activation)(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(fc2, tf.keras.layers.LeakyReLU(alpha=0.1),
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = tf.keras.layers.Activation(dense_activation)(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_dim, activation=output, name='predictions')(x)
    model = tf.keras.Model(inputs=input_tensor, outputs=x)

    return model

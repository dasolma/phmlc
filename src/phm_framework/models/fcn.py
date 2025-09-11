import tensorflow as tf
from phm_framework import typing
from phm_framework.trainers.net import NetTrainer

EXTRA_CHANNEL = False
TRAINER = NetTrainer

def create_model(input_shape, output, nhideen_layers: int = 2, activation: typing.Activation = tf.keras.layers.LeakyReLU(),
                 dropout: float = 0.5, l1: float = 1e-5, l2: float = 1e-4, output_dim: int = 1,
                 batch_normalization: bool = True):

    input_tensor = tf.keras.layers.Input(input_shape)
    x = input_tensor

    if input_shape[-2] > 5:
        x = tf.keras.layers.Permute((2, 1), input_shape=input_shape)(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(5))(x)
        x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Flatten()(x)

    neurons = x.shape[1]
    pows = []
    exp = 2
    while True:
        p = 2**exp
        if (p > output_dim) and (p < neurons):
            pows.append(p)

        if p > neurons:
            break

        exp += 1

    pows = pows[::-1]
    step = len(pows) / nhideen_layers

    for i, _ in enumerate(range(nhideen_layers)):

        neurons = pows[int(i*step)]
        if i == 0:
            neurons * 2

        x = tf.keras.layers.Dense(neurons, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2))(x)

        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation(activation)(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(output_dim, activation=output, name='predictions')(x)
    model = tf.keras.Model(inputs=input_tensor, outputs=x)

    return model

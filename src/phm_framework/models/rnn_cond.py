import tensorflow as tf
from phm_framework import typing
from phm_framework.trainers.net import NetTrainer

EXTRA_CHANNEL = False
TRAINER = NetTrainer

def create_model(input_shape, output, bidirectional: bool = True, cell_type: typing.RNNCell = tf.keras.layers.GRUCell,
                 nblocks: int = 3, rnn_units: int = 128, l1: float = 1e-4, l2: float = 1e-4,
                 dropout: float = 0.05, fc1: int = 128, fc2: int = 64,
                 dense_activation: typing.Activation = tf.keras.layers.ReLU(),
                 num_features: int = 0, batch_normalization: bool = True, output_dim: int = 1):
    # model creationç
    input_tensor = tf.keras.layers.Input(input_shape)
    x = input_tensor

    if input_shape[-1] == 1:
        input_shape = input_shape[:-1]
        x = tf.keras.layers.Reshape(input_shape)(x)

    if input_shape[-2] > 5:
        x = tf.keras.layers.Permute((2, 1), input_shape=input_shape)(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(5))(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Permute((2, 1), input_shape=input_shape)(x)

    rnn_units = [rnn_units] * nblocks
    for i, units in enumerate(rnn_units):
        # return_sequences = (i < len(rnn_units) - 1) or attention
        return_sequences = True
        cell = tf.keras.layers.RNN(cell_type(units=units,
                                             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2)),
                                   name='rnn_cell_%d' % (i + 1),
                                   return_sequences=return_sequences,
                                   )
        if bidirectional:
            x = tf.keras.layers.Bidirectional(cell)(x)
        else:
            x = cell(x)

        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)

    # FNN
    x = tf.keras.layers.Flatten()(x)


    if num_features > 0:
        input_features = tf.keras.layers.Input((num_features, 2))
        y = input_features

        #y = tf.keras.layers.Conv1D(1, (1,), activation='relu')(y)

        y = tf.keras.layers.Flatten()(y)
        y = tf.keras.layers.Dense(32, tf.keras.activations.swish)(y)
        y = tf.keras.layers.Dense(16, tf.keras.activations.swish)(y)
        #y = tf.keras.layers.Dense(8, dense_activation)(y)

        #y = tf.keras.layers.Dense(x.shape[1], dense_activation)(y)
        #x = tf.keras.layers.Attention()([x, y])
        x = tf.keras.layers.Concatenate()([x, y])


    x = tf.keras.layers.Dense(fc1,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = tf.keras.layers.Activation(dense_activation)(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(fc2,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = tf.keras.layers.Activation(dense_activation)(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_dim, activation=output, name='predictions')(x)

    if num_features > 0:
        model = tf.keras.Model(inputs=[input_tensor, input_features], outputs=x)
    else:
        model = tf.keras.Model(inputs=input_tensor, outputs=x)

    return model

import tensorflow as tf

class TransposeLast2ChannelsLayer(tf.keras.layers.Layer):
    """
    """

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, x):

        return tf.einsum("...ij->...ji", x)

    def get_config(self):
        config = {
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))



class ModelWrapper(tf.keras.Model):
    # Wrapper around an existing Keras model
    # Subclass of tf.keras.Model and can be treated as a standard Keras model.

    def __init__(self, model, *args, **kwargs):
        # Class constructor
        # Used to initialize the object’s attributes and perform any necessary setup when an instance of the class is created.
        # *args: variable number of positional arguments passed as a tuple.
        # A positional argument is passed to a function based on its position in the call.
        # They must be passed in the same order as defined in the function.
        # **kwargs: variable number of keyword arguments.
        # A keyword argument is passed to a function using the parameter name explicitly.
        # These arguments match the function's parameters by name, not position.
        # They can be passed in any order by specifying the name of the parameter.

        super().__init__(*args, **kwargs)  # Initialize the base class

        self.base_model = model  # Assign the model to the base_model attribute of the current object (self)

    @property  # Define methods as properties so they can be accessed like attributes
    def metrics_names(self):
        # Return the names of the metrics from the base model
        return self.base_model.metrics_names

    @property
    def metrics(self):
        # Return the actual metric objects in a list
        return self.base_model.metrics

    def compile(self, *args, **kwargs):
        # Compile the model
        return self.base_model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        # Train the model
        return self.base_model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        # Evaluate the model’s performance
        return self.base_model.evaluate(*args, **kwargs)

    def summary(self, *args, **kwargs):
        # Print a summary of the model
        return self.base_model.summary(*args, **kwargs)

    def save(self, *args, **kwargs):
        # Save the model for future use
        return self.base_model.save(*args, **kwargs)

    def call(self, *args, **kwargs):
        # Call the model to perform inference or a forward pass with the input data
        return self.base_model.call(*args, **kwargs)



def create_encoder(input_shape, filters, batch_normalization, conv_activation, l1, l2, nblocks, block_size, embedding_dim):
    # -- begin encoder
    input_tensor = tf.keras.layers.Input(input_shape)
    x = input_tensor

    # WDCNN
    #x = tf.keras.layers.Conv1D(filters=filters, kernel_size=64, strides=16, activation='relu', padding='same',
    #                           input_shape=input_shape, name="encoder_conv_1")(input_tensor)
    #x = tf.keras.layers.Activation(conv_activation)(x)
    #x = tf.keras.layers.MaxPooling1D(strides=2, name=f"encoder_maxpool_1")(x)
    for i in range(nblocks):
        filters = min(64, filters * 2)
        strides = 2 if i == 0 else 1
        kernel_size = 3 if i == 0 else 2

        for j in range(block_size):
            x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                                       kernel_initializer='he_uniform',
                                       strides=strides, activation='relu', padding='same',
                                       name=f"encoder_conv_{i + 2}.{j}")(x)
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Activation(conv_activation, name=f"encoder_conv_act_{i + 2}.{j}")(x)

        if x.shape[1] > 4:
            x = tf.keras.layers.MaxPooling1D(strides=2, name=f"encoder_maxpool_{i + 2}")(x)

    if embedding_dim is not None:
        x = tf.keras.layers.Flatten(name="encoder_flatten_1")(x)
        x = tf.keras.layers.Dense(embedding_dim, activation='linear', name="embedding")(x)

    embedding = tf.keras.Model(inputs=input_tensor, outputs=x)

    return embedding

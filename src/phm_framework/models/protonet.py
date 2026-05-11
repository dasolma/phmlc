import tensorflow as tf
from phm_framework.models.base import create_encoder

'''
Prototypical Network: Protonet
    All samples from the support and query sets are used simultaneously
    1 - Feature extraction from the support and query sets and obtaining their 
    respective vector representations or embeddings
    2 - For each class in the support set, the mean of the embeddings corresponding 
    to that class is calculated, generating a "prototype" for each class
    3 - The distance between a query embedding and a prototype is measured using Euclidean distance
'''

EXTRA_CHANNEL = True


class DistanceLayer(tf.keras.layers.Layer):
    """
    Compute distances between input vectors
    """

    def __init__(self, distance_type='euclidean', *args, **kwargs):  # Class constructor
        # Initializes the layer with a default distance type, which is 'euclidean'.
        super().__init__(*args, **kwargs)

        self.distance_type = distance_type

    def call(self, q, s):
        # Method that performs the distance computation between the input tensors q and s
        if self.distance_type == 'l1':  # Manhattan distance
            distance = tf.reduce_sum(tf.math.abs(q - s),
                                     axis=-1)  # Sums the absolute values along the last axis of the resulting tensor
            return distance

        elif self.distance_type == 'l2':  # Euclidean distance
            distance = tf.reduce_sum(tf.square(q - s), axis=-1)
            return distance

        elif self.distance_type == 'cosine':  # Cosine similarity
            normalize_a = tf.math.l2_normalize(q, -1)  # Normalize the vectors to the same scale, unit vectors
            normalize_b = tf.math.l2_normalize(s, -1)
            cos_similarity = -tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=-1)
            # The dot product of two normalized unit vectors is equal to the cosine of the angle between them
            # Negative cosine similarity
            return cos_similarity

        elif self.distance_type == 'product':  # Dot product method
            # Useful when the goal is to compute interaction between features represented by the vectors
            return -tf.reduce_sum(tf.multiply(q, s), axis=-1)

    def get_config(self):
        config = {
            "distance_type": self.distance_type,
        }  # Dictionary with the distance type used to initialize the DistanceLayer
        base_config = super().get_config()  # Base configuration from the superclass
        return dict(list(base_config.items()) + list(
            config.items()))  # Dictionary containing both specific DistanceLayer config and base config


class PrototypeEmbedding(tf.keras.layers.Layer):
    '''
    Computes the centroid of a set of input data along a specific dimension
    Returns this centroid as its output
    '''

    def __init__(self, **kwargs):  # Class constructor
        super().__init__(**kwargs)  # Base class constructor

    def call(self, x):
        # Input centroids:
        x = tf.reduce_mean(x, axis=-1)  # Mean along the last axis
        # Mean of all features, producing a single centroid vector for each instance

        # x = tf.nn.leaky_relu(x)

        output = x
        return output  # Centroids computed as the output of the layer


class WDCNNProtonet(tf.keras.Model):
    '''
    Class that implements Protonet using a deep convolutional network architecture (WDCNN)
    '''

    def __init__(self, input_shape, filters, batch_normalization, conv_activation, l1, l2, nblocks, block_size, nlabels,
                 embedding_dim, distance, base_model):
        '''Class constructor'''
        super(WDCNNProtonet, self).__init__()

        self._input_shape = input_shape  # Input shape of the data
        self.filters = filters  # Initial number of filters for the convolutional layers
        self.batch_normalization = batch_normalization  # Boolean indicating whether batch normalization is applied after the convolutional layers
        self.conv_activation = conv_activation  # Activation function for the convolutional layers
        self.l1 = l1  # L1 regularization parameter
        self.l2 = l2  # L2 regularization parameter
        self.nblocks = nblocks  # Number of convolutional blocks in the network
        self.nlabels = nlabels  # Number of labels or classes
        self.filters = filters  # Number of filters for the convolutional layers
        self.embedding_dim = embedding_dim  # Dimension of the embedding space
        self.distance = distance  # Type of distance function or layer to use

        # Instance of the model that encodes the input data to an embedding space:
        if base_model is None:
            self.encoder = create_encoder(input_shape, filters, batch_normalization, conv_activation, l1, l2, nblocks,
                                          block_size, embedding_dim)
        else:
            self.encoder = base_model

        # Layer instance that computes the distance between support and query embeddings:
        self.distance = DistanceLayer(distance)

    def summary(self, *args, **kwargs):
        ''' Method that prints a summary of the encoder model '''
        self.encoder.summary(*args, **kwargs)

    def __call__(self, data, *args, **kwargs):
        Q, S, SY = data
        Qe, Se = self.get_embeddings((Q, S))

        # To compute the distances between each query embedding and support embeddings:
        # Repeat Qe along the first axis to match the number of support examples (Se)
        Qe = tf.repeat(Qe, tf.shape(S)[0], axis=0)

        Se = tf.repeat([Se], tf.shape(Q)[0], axis=0)
        Se = tf.reshape(Se, (tf.shape(Se)[0] * tf.shape(Se)[1], tf.shape(Se)[2]))

        # Distances between query embeddings (Qe) and support embeddings (Se)
        d = -self.distance(Qe, Se)

        # Reshape distances to (number_of_queries, number_of_shots)
        d = tf.reshape(d, (tf.shape(Q)[0], tf.shape(S)[0]))

        # Probabilities: normalize predictions to be in [0, 1] and sum to 1
        w = tf.nn.softmax(d, axis=-1)

        power = 3.0  # Puedes probar con 2.0, 3.0, etc.
        w_accentuated = tf.pow(w, power)

        # Normalizar nuevamente para que sigan sumando 1 por fila
        w_accentuated /= tf.reduce_sum(w_accentuated, axis=-1, keepdims=True)

        # Uso de los nuevos pesos acentuados
        preds = tf.reduce_sum(SY * w_accentuated, axis=-1)

        # preds = tf.reduce_sum(SY * w, axis=-1)

        return preds

    def get_embeddings(self, data):
        Q, S = data
        # create embeddings of the queries and support set
        # Data encoding: vector representations in the embedding space:
        Qe = self.encoder(Q)

        # Sprime = tf.reshape(S, (tf.shape(S)[0] * tf.shape(S)[1], tf.shape(S)[2], tf.shape(S)[3]))
        # Se = self.encoder(Sprime)  # Support set embeddings

        Se = self.encoder(S)

        return Qe, Se
        # Qe: query set embeddings
        # Se: support set embeddings

    def get_testmodel(self, num_labels, k):
        return self


def create_model(input_shape, output_dim: int,
                 output,
                 embedding_dim: int = 256,
                 nblocks: int = 3,
                 block_size: int = 3,
                 filters: int = 16,
                 l1: float = 0,
                 l2: float = 0,
                 distance='l2',
                 conv_activation=tf.keras.layers.ReLU(),
                 base_model=None,
                 batch_normalization: bool = True):
    if base_model is not None:
        base_model, space_dims = base_model

    return WDCNNProtonet(input_shape, filters, batch_normalization, conv_activation, l1, l2, nblocks, block_size,
                         output_dim, embedding_dim, distance, base_model)

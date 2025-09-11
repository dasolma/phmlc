"""

Adapted from: https://github.com/aqibsaeed/Sensor-Transformer/blob/main/sensortransformer/set_network.py,
which is an adaptation of the vision transformer: https://arxiv.org/pdf/2010.11929.pdf

"""
import math

import tensorflow as tf
from einops.layers.tensorflow import Rearrange
import numpy as np
from phm_framework.trainers.net import NetTrainer

EXTRA_CHANNEL = False
TRAINER = NetTrainer

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation=tf.keras.activations.gelu),
             tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm_a = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_b = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_a = tf.keras.layers.Dropout(dropout)
        self.dropout_b = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout_a(attn_output,
                                     training=training)
        out_a = self.layernorm_a(inputs + attn_output)
        ffn_output = self.ffn(out_a)
        ffn_output = self.dropout_b(ffn_output,
                                    training=training)
        return self.layernorm_b(out_a + ffn_output)

    def get_config(self):
        config = {"embed_dim": self.embed_dim,
                  "num_heads": self.num_heads,
                  "ff_dim": self.ff_dim,
                  "dropout": self.dropout
                  }
        base_config = super(TransformerBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len=None, embedding_dim=None, **kwargs):
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        super(PositionalEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        if self.embedding_dim == None:
            self.embedding_dim = int(inputs.shape[-1])

        position_embedding = np.array([
            [pos / np.power(10000, 2. * i / self.embedding_dim) for i in range(self.embedding_dim)]
            for pos in range(self.sequence_len)])

        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])  # dim 2i
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])  # dim 2i+1
        position_embedding = np.expand_dims(position_embedding,axis=0)
        position_embedding = tf.cast(position_embedding, dtype=tf.float32)

        return position_embedding

    def get_config(self):
        config = {"sequence_len": self.sequence_len, "embedding_dim": self.embedding_dim}
        base_config = super(PositionalEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SensorTransformer(tf.keras.Model):
    def __init__(
            self,
            signal_length,
            segment_size,
            num_layers,
            d_model,
            num_heads,
            mlp_dim,
            dropout,
    ):
        super(SensorTransformer, self).__init__()
        num_patches = (signal_length // segment_size)

        self.signal_length = signal_length
        self.segment_size = segment_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.pos_emb = self.add_weight("pos_emb",
                                       shape=(1, num_patches + 1, d_model))
        self.class_emb = self.add_weight("class_emb",
                                         shape=(1, 1, d_model))
        self.patch_proj = tf.keras.layers.Dense(d_model)
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]
        self.mlp_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(mlp_dim,
                                      activation=tf.keras.activations.gelu),
            ]
        )

    def call(self, input, training):
        batch_size = tf.shape(input)[0]
        patches = Rearrange("b c (w p1) -> b w (c p1)",
                            p1=self.segment_size)(input)
        x = self.patch_proj(patches)

        class_emb = tf.broadcast_to(self.class_emb,
                                    [batch_size, 1, self.d_model])

        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x, training)

        return self.mlp_head(x[:, 0])

    def get_config(self):
        config = {
            'signal_length': self.signal_length,
            'segment_size': self.segment_size,
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'mlp_dim': self.mlp_dim,
            'dropout': self.dropout,
        }

        return config


def create_model(input_shape, output, nlayers: int = 3, segment_size: float = 0.05,
                 model_dim: int = 32, num_heads: int = 32, dropout: float = 0.05,
                 mlp_dim: int = 128, output_dim: int = 1):
    signal_length = input_shape[-1]

    divs = [i for i in range(1, signal_length//2) if signal_length % i == 0]
    segment_size = max(1, int(signal_length * segment_size))
    segment_size = min(divs, key=lambda x: abs(x-segment_size))

    input = tf.keras.Input(input_shape)
    x = input

    if input_shape[-2] > 5:
        x = tf.keras.layers.Permute((2, 1), input_shape=input_shape)(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(5))(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Permute((2, 1), input_shape=input_shape)(x)

    transformer = SensorTransformer(signal_length, segment_size, nlayers,
                              model_dim, num_heads, mlp_dim, dropout)(x)

    x = tf.keras.layers.Dense(output_dim, activation=output, name='predictions')(transformer)

    model = tf.keras.Model(inputs=input, outputs=x)


    return model
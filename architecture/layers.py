import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import unit_norm

from GeodesicRelaxationDCCA.algorithms.losses_metrics import compute_l1, compute_l2


class Encoder(tf.keras.Model):
    def __init__(self, config, view_ind, **kwargs):
        super(Encoder, self).__init__(name=f'Encoder_view_{view_ind}', **kwargs)
        self.config = config
        self.view_index = view_ind

        self.dense_layers = {
            str(i): layers.Dense(
                dim,
                activation=activ,
            ) for i, (dim, activ) in enumerate(self.config)
        }

    def get_l1(self):
        return tf.math.reduce_sum([compute_l1(layer.trainable_variables[0]) for id, layer in self.dense_layers.items()])

    def get_l2(self):
        return tf.math.reduce_sum([compute_l2(layer.trainable_variables[0]) for id, layer in self.dense_layers.items()])

    def call(self, inputs):
        x = inputs
        for i in range(len(self.dense_layers)):
            x = self.dense_layers[str(i)](x)

        return x



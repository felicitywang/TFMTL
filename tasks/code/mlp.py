# Copyright 2017 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def trunc_normal(stddev):
    return tf.truncated_normal_initializer(stddev=stddev)


class MLP(object):
    def __init__(self,
                 x,  # Batch of examples: [batch_size, feature_size]
                 y,  # Batch of targets: [batch_size]
                 num_classes,  # Number of classes
                 keep_prob=0.5,
                 layers=[512, 256, 128],
                 l2_weight=0.001,
                 is_training=True,
                 name='MLP'):

        with tf.name_scope(name=name):
            self._batch_size = tf.shape(x)[0]
            self._targets = y

            # MLP
            for layer in layers:
                if is_training:
                    x = tf.contrib.layers.dropout(x, keep_prob)
                x = tf.contrib.layers.fully_connected(
                    x,
                    layer,
                    weights_regularizer=tf.contrib.layers.l2_regularizer(
                        l2_weight))

            # Output projection
            x = tf.contrib.layers.fully_connected(
                x,
                num_classes,
                weights_initializer=trunc_normal(1. / float(layers[-1])),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)

            # Loss
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                logits=x)
            batch_size = tf.cast(self._batch_size, tf.float32)
            self._loss = tf.reduce_mean(ce)

            # Accuracy
            x = tf.nn.softmax(x)
            y_hat = tf.argmax(x, 1)
            self._correct = tf.reduce_sum(
                tf.cast(tf.equal(y_hat, y), tf.float32))
            self._accuracy = self._correct / batch_size

    @property
    def loss(self):
        return self._loss

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def num_correct_predicted(self):
        return self._correct

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def targets(self):
        return self._targets

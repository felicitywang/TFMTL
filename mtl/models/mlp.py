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
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class MLP(object):
    def __init__(self,
                 inputs,  # Batch of examples: [batch_size, feature_size]
                 labels,  # Batch of targets: [batch_size]
                 num_classes,  # Number of classes
                 dropout_rate=0.5,
                 layers=[512, 256, 128],
                 # l2_weight=0.001,
                 l2_weight=0.0,
                 is_training=True,
                 name='MLP'):
        with tf.name_scope(name=name):
            self._batch_size = tf.shape(inputs)[0]
            self._targets = labels

            for layer in layers:
                inputs = tf.layers.dense(inputs=inputs, units=layer,
                                         activation=tf.nn.relu)
            dropout = tf.layers.dropout(inputs=inputs, rate=dropout_rate,
                                        training=is_training)
            logits = tf.layers.dense(inputs=dropout, units=num_classes,
                                     activation=tf.nn.relu)
            predicted_classes = tf.argmax(input=logits, axis=1)

            # loss
            ce = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=tf.cast(labels, dtype=tf.int32)))

            # l2 regularization
            variables = tf.trainable_variables()
            l2_weight_penalty = tf.add_n([tf.nn.l2_loss(v) for v in variables
                                          if 'bias' not in v.name]) * l2_weight
            self._loss = tf.reduce_mean(ce) + l2_weight_penalty

            # Accuracy
            self._correct = tf.reduce_sum(
                tf.cast(tf.equal(predicted_classes, labels), tf.float32))
            self._accuracy = 1.0 * self._correct / tf.cast(self._batch_size,
                                                           tf.float32)

    @property
    def loss(self):
        return self._loss

    @property
    def accuracy(self):
        return self._accuracy

    @property
    # number of correct predictions
    def correct(self):
        return self._correct

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def targets(self):
        return self._targets

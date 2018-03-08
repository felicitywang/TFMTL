# Copyright 2018 Johns Hopkins University. All Rights Reserved.
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
from six.moves import xrange
from tensorflow.python.ops.init_ops import glorot_uniform_initializer
from tensorflow.python.ops.init_ops import zeros_initializer

from mtl.hparams import get_activation_fn


def dense_layer(x, output_size, name, activation=tf.nn.selu):
    assert type(output_size) is int
    assert type(name) is str

    if type(activation) == str:
        activation = get_activation_fn(activation)

    if activation == tf.nn.selu:
        init = tf.variance_scaling_initializer(scale=1.0, mode='fan_in')
    else:
        init = glorot_uniform_initializer()

    return tf.layers.dense(x, output_size, name=name,
                           kernel_initializer=init,
                           bias_initializer=zeros_initializer(),
                           activation=activation)


def mlp(x, is_training, output_size, hidden_dim=256, num_layer=2, activation=tf.nn.selu,
        input_keep_prob=1.0, batch_normalization=False,
        layer_normalization=True, output_keep_prob=1.0
        ):
    if num_layer < 1:
        return x
    if activation == tf.nn.selu:
        dropout = tf.contrib.nn.alpha_dropout
    else:
        dropout = tf.nn.dropout

    if input_keep_prob < 1.0:
        x = dropout(x, input_keep_prob, name='input_dropout')

    assert not (batch_normalization and layer_normalization)
    assert not (hidden_dim is None)
    assert not (num_layer is None)

    for i in xrange(num_layer):
        with tf.variable_scope("layer_%d" % i):
            x = dense_layer(x, hidden_dim, 'linear', activation=None)
            if batch_normalization:
                x = tf.layers.batch_normalization(x, training=is_training,
                                                  name='bn')
            if layer_normalization:
                x = tf.contrib.layers.layer_norm(x, name='ln')
            x = activation(x)

    if output_keep_prob < 1.0:
        x = dropout(x, output_keep_prob, name='output_dropout')

    return x

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

import mtl.util.registry as registry


def encode(inputs, lengths, is_training,
           encoder='simple_birnn',
           hparams='simple_birnn_default',
           embed_fn=None,
           embed_dim=None,
           embed_l2_scale=0.0,
           initializer_stddev=0.001):
    # Project integer inputs to vectors via an embedding
    if embed_fn is None:
        assert embed_dim is not None
        regularizer = None
        if embed_l2_scale > 0.0:
            regularizer = tf.contrib.layers.l2_regularizer(embed_l2_scale)
        initializer = tf.truncated_normal_initializer(mean=0.0,
                                                      stddev=initializer_stddev)
        with tf.variable_scope("input_embedding", reuse=tf.AUTO_REUSE):
            embed_matrix = tf.get_variable("embed_matrix",
                                           [vocab_size, embed_dim],
                                           regularizer=regularizer,
                                           initializer=initializer)
        x = tf.nn.embedding_lookup(embed_matrix, inputs)
    else:
        x = embed_fn(inputs)

    # Encode inputs, collapsing over axis=1
    encoder_fn = registry.encoder(encoder)
    hp = registry.hparams(hparams)
    code = encoder_fn(x,
                      lengths,
                      hp=hp(),
                      is_training=is_training)

    assert len(code.get_shape().as_list()) == 2
    return code

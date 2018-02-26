# Copyright 2017 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.seq2seq import sequence_loss


def ngram(x, z, vocab_size, embed_dim, ngram_order=3):
    # Unpack observations
    if len(x) != 3:
        raise ValueError('expected x to contain (targets, contexts, lens)')

    targets, contexts, lens = x
    targets_shape = tf.shape(targets)
    batch_size = targets_shape[0]
    batch_len = targets_shape[1]

    # Embed context word IDs
    contexts = tf.reshape(contexts, [batch_size, batch_len, ngram_order])
    with tf.variable_scope("embedding"):
        embed = tf.contrib.layers.embed_sequence(contexts, vocab_size=vocab_size,
                                                 embed_dim=embed_dim)

    # Join N-Gram Embeddings with Latent Code
    unstacked = tf.unstack(embed, axis=2)
    tiled_z = tf.expand_dims(z, 1)
    tiled_z = tf.tile(tiled_z, [1, batch_len, 1])
    joined = tf.concat(unstacked + [tiled_z], axis=-1)

    with tf.variable_scope("output_projection"):
        logits = tf.layers.dense(joined, vocab_size, activation=tf.nn.elu)

    # Get sequence mask as 0/1 weights
    with tf.name_scope("weights"):
        weights = tf.to_float(tf.sequence_mask(lens, maxlen=batch_len))

    # Compute reconstruction error (average over time). Note that this
    # masks by multiplying the cross-entropy by the sequence mask as 0/1
    # weights.
    with tf.name_scope("sequence_loss"):
        losses = sequence_loss(logits, targets, weights,
                               average_across_batch=False,
                               average_across_timesteps=False)

        return tf.reduce_sum(losses, axis=1)

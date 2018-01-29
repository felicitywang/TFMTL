# Copyright 2018 Johns Hopkins University. All Rights Reserved.
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


def unigram(x, z, vocab_size):
  """ This implements the unigram output projection from:

    Neural Variational Inference for Text Processing
    Yishu Miao, Lei Yu, Phil Blunsom
    https://arxiv.org/abs/1511.06038

  """
  # Unpack observations
  if len(x) != 3:
    raise ValueError('expected x to contain (targets, counts, lens)')

  targets, counts, lens = x

  targets_shape = targets.get_shape()
  if len(targets_shape) != 2:
    raise ValueError("expected 2D targets: got %d" % (len(targets_shape)))
  counts_shape = counts.get_shape()
  if len(counts_shape) != 2:
    raise ValueError("expected 2D counts: got %d" % (len(counts_shape)))

  input_dim = z.get_shape().as_list()[-1]
  target_len = tf.shape(targets)[1]
  with tf.variable_scope("output_projection"):
    R = tf.get_variable(
      "R",
      [input_dim, vocab_size],
      initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(
      "b",
      [vocab_size],
      initializer=tf.zeros_initializer())

  # Compute logits & shape for sequence loss
  with tf.name_scope("logits"):
    logits = tf.nn.xw_plus_b(z, R, b)
    logits = tf.expand_dims(logits, 1)
    logits = tf.tile(logits, [1, target_len, 1])

  # Get sequence mask as 0/1 weights
  with tf.name_scope("weights"):
    weights = tf.to_float(tf.sequence_mask(lens, maxlen=target_len))

  # Compute reconstruction error (average over time).  Note that this
  # masks by multiplying the crossent by the sequence mask as 0/1
  # weights.
  with tf.name_scope("sequence_loss"):
    losses = sequence_loss(logits, targets, weights,
                           average_across_batch=False,
                           average_across_timesteps=False)

    # Scale the losses by the number of observations, then average
    # across time steps. These are negative *log* probabilities, so we
    # multiply by the count.
    if counts is not None:
      losses *= tf.to_float(counts)

    return tf.reduce_sum(losses, axis=1)

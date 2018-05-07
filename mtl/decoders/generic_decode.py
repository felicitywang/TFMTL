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
from tensorflow.contrib.seq2seq import sequence_loss
from mtl.layers.t2t import conv_wn
import mtl.util.registry as registry


def shift_right(x, pad_value=None):
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0]])[:, :-1]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1]
  return shifted_targets


def decode(targets, lengths, vocab_size, is_training,
           global_conditioning=None, decoder='resnet',
           hparams='resnet_default', embed_fn=None, embed_dim=None,
           embed_l2_scale=0.0, initializer_stddev=0.001, add_timing=False,
           average_across_timesteps=False, average_across_batch=False):
  # Shift targets right to get inputs
  inputs = shift_right(targets)

  print('inputs:')
  print(inputs)
  print('targets:')
  print(targets)

  # Project integer inputs to vectors via an embedding
  if embed_fn is None:
    assert embed_dim is not None
    regularizer = None
    if embed_l2_scale > 0.0:
      regularizer = tf.contrib.layers.l2_regularizer(embed_l2_scale)
    initializer = tf.truncated_normal_initializer(mean=0.0,
                                                  stddev=initializer_stddev)
    with tf.variable_scope("input_embedding", reuse=tf.AUTO_REUSE):
      embed_matrix = tf.get_variable("embed_matrix", [vocab_size, embed_dim],
                                     regularizer=regularizer,
                                     initializer=initializer)
    x = tf.nn.embedding_lookup(embed_matrix, inputs)
  else:
    x = embed_fn(inputs)

  print('Embedded inputs:')
  print(x)

  decoder_fn = registry.decoder(decoder)

  # Get predictions for each target
  #assert_op = tf.Assert(tf.greater_equal(tf.shape(x)[1], 1), [x])
  #with tf.control_dependencies([assert_op]):
  if True:
    if 'rnn' in decoder:
      x = decoder_fn(x, is_training, hp=registry.hparams(hparams),
                     global_conditioning=global_conditioning)
      logits = tf.layers.dense(x, vocab_size, use_bias=False)
    else:
      x = tf.expand_dims(x, axis=2)
      x = decoder_fn(x, is_training, hp=registry.hparams(hparams),
                     global_conditioning=global_conditioning)
      k = (1, 1)
      x = conv_wn(x, vocab_size, k, padding='LEFT')
      logits = tf.squeeze(x, axis=2)

  # Mask for variable length targets
  batch_size = tf.shape(targets)[0]
  batch_len = tf.shape(targets)[1]
  if lengths is None:
    mask = tf.ones([batch_size, batch_len])
  else:
    mask = tf.to_float(tf.sequence_mask(lengths, maxlen=batch_len))

  # Compute loss
  loss = sequence_loss(
    logits, targets, mask, average_across_timesteps=average_across_timesteps,
    average_across_batch=average_across_batch)

  return loss

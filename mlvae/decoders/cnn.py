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
from tensorflow.contrib.training import HParams
from mlvae.t2t import conv1d


def default_hparams():
  return HParams(filter_width=3,
                 embed_dim=32,
                 num_filters=128)


def shift_right(x, pad_value=None):
  # TODO(noa): test this

  """Shift the 2nd dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :]
  return shifted_targets


def cnn(batch, z, vocab_size, embedder=None, hp=default_hparams(),
        targets_key="tokens", lengths_key="tokens_length"):
  targets = batch[targets_key]
  lengths = batch[lengths_key]
  assert len(targets.get_shape().as_list()) == 2
  assert len(lengths.get_shape().as_list()) == 1
  targets_shape = tf.shape(targets)
  batch_len = targets_shape[1]
  if embedder is None:
    tf.logging.info("[CNN decoder] Using new word embedding.")
    x = tf.contrib.layers.embed_sequence(targets, vocab_size=vocab_size,
                                         embed_dim=hp.embed_dim)
  else:
    tf.logging.info("[CNN decoder] Using existing word embedder.")
    x = embedder(targets)
  x = shift_right(x)
  x = conv1d(x, hp.num_filters, hp.filter_width,
             dilation_rate=1, use_bias=False, padding='CAUSAL',
             name='conv1d_pre')
  z = tf.expand_dims(z, 1)
  z_tile = tf.tile(z, [1, batch_len, 1])
  xz = tf.concat([x, z_tile], axis=2)
  xz = tf.nn.relu(xz)
  xz = conv1d(
    xz, hp.num_filters, 1, dilation_rate=1,
    use_bias=True, padding='CAUSAL', name='conv1d_post1')
  xz = tf.nn.relu(xz)
  logits = conv1d(
    xz, vocab_size, 1, dilation_rate=1,
    use_bias=True, padding='CAUSAL', name='conv1d_post2')
  mask = tf.to_float(tf.sequence_mask(lengths, maxlen=batch_len))
  losses = sequence_loss(logits, targets, mask,
                         average_across_timesteps=False,
                         average_across_batch=False)
  return tf.reduce_sum(losses, axis=1)

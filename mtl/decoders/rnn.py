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
from six.moves import xrange


@registry.register_hparams
def rnn_default():
  hps = tf.contrib.training.HParams(
    cell_type = 'gru',
    cell_size = 512,
    num_layer = 2,
    keep_prob = 0.75
  )
  return hps


def rnn_cell(name, hp):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    if hp.cell_type == 'gru':
      cell = tf.nn.rnn_cell.GRUCell(hp.cell_size)
    elif hp.cell_type == 'lstm':
      cell = tf.contrib.rnn.BasicLSTMCell(hp.cell_size)
    elif hp.cell_type == 'fused_lstm':
      cell = tf.contrib.rnn.LSTMBlockFusedCell(hp.cell_size)
    elif hp.cell_type == 'block_lstm':
      cell = tf.contrib.rnn.LSTMBlockCell(hp.cell_size)
    elif hp.cell_type == 'ln_lstm':
      cell = tf.contrib.rnn.LayerNormBasicLSTMCell(hp.cell_size)
    else:
      raise ValueError('unrecognized cell type: %s' % (hp.cell_type))
  return cell


@registry.register_decoder
def rnn(x, is_training, global_conditioning=None,
        hp=rnn_default(), name='rnn'):
  ndims = len(x.get_shape().as_list())
  if ndims != 3:
    raise ValueError("expected 3D input, got %dD" % ndims)
  hp = hp()
  with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
    cells = []
    for l in xrange(hp.num_layer):
      cell = rnn_cell('cell_%d' % (l), hp)
      if is_training and hp.keep_prob < 1.0:
        cell = tf.nn.rnn_cell.DropoutWrapper(
          cell, output_keep_prob=hp.keep_prob)
      cells += [cell]

    cell = tf.contrib.rnn.MultiRNNCell(cells)

  batch_size = tf.shape(x)[0]
  batch_len = tf.shape(x)[1]

  if global_conditioning is not None:
    tf.logging.info("[RNNDecoder] Adding global conditioning")
    h = global_conditioning
    h = tf.expand_dims(h, axis=1)
    h = tf.tile(h, [1, batch_len, 1])
    x = tf.concat([x, h], axis=-1)

  initial_state = cell.zero_state(batch_size, tf.float32)
  outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=x,
                                 sequence_length=None,
                                 initial_state=initial_state,
                                 time_major=False)


  return outputs

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


def stacked_rnn_cell(num_layer, cell_type, cell_size, keep_prob=1.0,
                     scope="stacked_rnn_cell", **kwargs):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    cells = []
    for l in xrange(num_layer):
      with tf.variable_scope("cell_%d" % l):
        if cell_type == 'lstm':
          cell = tf.contrib.rnn.LSTMBlockCell(cell_size, **kwargs)
        else:
          raise ValueError('unrecognized cell type: %s' % cell_type)
        if keep_prob < 1.0:
          if l == 0:
            cell = tf.nn.rnn_cell.DropoutWrapper(
              cell,
              input_keep_prob=keep_prob,
              output_keep_prob=keep_prob)
          else:
            cell = tf.nn.rnn_cell.DropoutWrapper(
              cell,
              output_keep_prob=keep_prob)
        cells += [cell]
    return tf.contrib.rnn.MultiRNNCell(cells)

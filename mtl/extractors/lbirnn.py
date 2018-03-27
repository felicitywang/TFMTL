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

import tensorflow as tf
from six.moves import xrange

from mtl.util.reducers import (reduce_max_over_time,
                               reduce_avg_over_time,
                               reduce_var_over_time,
                               reduce_min_over_time)


def get_multi_cell(cell_type, cell_size, num_layers):
  cells = [cell_type(cell_size) for _ in xrange(num_layers)]
  return tf.contrib.rnn.MultiRNNCell(cells)


def lbirnn_and_pool(inputs,
                    lengths,
                    num_layers=2,
                    cell_type=tf.contrib.rnn.BasicLSTMCell,
                    cell_size=64,
                    initial_state_fwd=None,
                    initial_state_bwd=None,
                    reducer=reduce_max_over_time):
  """Stacked linear chain bi-directional LSTM"""
  # TODO: determine dimensions of output representation

  # TODO: does padding affect how we want to do the reversal?

  reducers = [reduce_avg_over_time,
              reduce_var_over_time,
              reduce_max_over_time,
              reduce_min_over_time]
  assert reducer in reducers, "unrecognized bi-rnn reducer: %s" % reducer

  inputs_rev = tf.reverse(inputs, [1])  # reverse along time axis

  cells_fwd = get_multi_cell(cell_type, cell_size, num_layers)
  cells_bwd = get_multi_cell(cell_type, cell_size, num_layers)

  batch_size = tf.shape(inputs)[0]
  if initial_state_fwd is None:
    initial_state_fwd = cells_fwd.zero_state(batch_size,
                                             tf.float32)
  if initial_state_bwd is None:
    initial_state_bwd = cells_bwd.zero_state(batch_size,
                                             tf.float32)

  outputs_fwd, states_fwd = tf.nn.dynamic_rnn(cells_fwd,
                                              inputs,
                                              sequence_length=lengths,
                                              initial_state=initial_state_fwd,
                                              time_major=False,
                                              scope="rnn_fwd")

  # TODO: does "lengths" have to change due to padding and reversal of inputs?
  outputs_bwd, states_bwd = tf.nn.dynamic_rnn(cells_bwd,
                                              inputs_rev,
                                              sequence_length=lengths,
                                              initial_state=initial_state_bwd,
                                              time_major=False,
                                              scope="rnn_bwd")

  outputs = tf.concat([outputs_fwd, outputs_bwd], axis=2)

  # Pooling
  return reducer(outputs, lengths=lengths, time_axis=1)

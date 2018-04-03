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


def get_multi_cell(cell_type, cell_size, num_layers):
  cells = [cell_type(cell_size) for _ in xrange(num_layers)]
  return tf.contrib.rnn.MultiRNNCell(cells)


def lbirnn(inputs,
           lengths,
           indices=None,
           num_layers=2,
           cell_type=tf.contrib.rnn.BasicLSTMCell,
           cell_size=64,
           initial_state_fwd=None,
           initial_state_bwd=None):
  """Stacked linear chain bi-directional RNN

  Inputs
  _____
    inputs: batch of size [batch_size, batch_len, embed_size]
    lengths: batch of size [batch_size]
    num_layers: number of stacked layers in the bi-RNN
    cell_type: type of RNN cell to use (e.g., LSTM, GRU)
    cell_size: cell's output size
    initial_state_fwd: initial state for forward direction
    initial_state_bwd: initial state for backward direction
    indices: which token index in each batch example should be output
             shape: [batch_size] or [batch_size, 1]

  Outputs
  _______
    If the input word vectors have dimension D and indices is None,
    the output is a Tensor of size
      [batch_size, batch_len, cell_size_fwd + cell_size_bwd]
        = [batch_size, batch_len, 2*cell_size].

    If indices is not None, the output is a Tensor of size
      [batch_size, cell_size_fwd + cell_size_bwd]
        = [batch_size, 2*cell_size]
  """

  # TODO: does padding affect how we want to do the reversal?
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

  if indices is not None:
    # row index [[0], [1], ..., [N]]
    r = tf.range(batch_size)
    r = tf.expand_dims(r, 1)

    # make sure indices are able to be concatenated with range
    # i.e., of the form [[idx_0], [idx_1], ..., [idx_N]]
    rank = len(indices.get_shape().as_list())
    if rank == 1:
      indices = tf.expand_dims(indices, 1)
    elif rank == 2:
      pass
    else:
      raise ValueError("indices doesn't have rank 1 or 2: rank=%d" % (rank))

    idx = tf.concat([r, indices], axis=1)

    # get the (indices[i])-th token's output from row i
    outputs = tf.gather_nd(outputs, idx)

  return outputs

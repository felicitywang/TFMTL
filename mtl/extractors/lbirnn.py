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
from six.moves import xrange
import mtl.util.registry as registry


def get_multi_cell(cell_type, cell_size, num_layers):
  if cell_type == tf.contrib.rnn.GRUCell:
    cell = cell_type(cell_size,
                     kernel_initializer=tf.contrib.layers.xavier_initializer())

  elif cell_type == tf.contrib.rnn.LSTMCell:
    cell = cell_type(cell_size,
                     initializer=tf.contrib.layers.xavier_initializer())

  else:
    cell = cell_type(cell_size)

  if num_layers > 1:
    return tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
  else:
    return cell


def _lbirnn_helper(inputs,
                   lengths,
                   is_training,
                   indices=None,
                   num_layers=2,
                   cell_type=tf.contrib.rnn.LSTMCell,
                   cell_size=64,
                   initial_state_fwd=None,
                   initial_state_bwd=None,
                   scope=None,
                   **kwargs):
  """Stacked linear chain bi-directional RNN

  Inputs
  _____
    inputs: batch of size [batch_size, batch_len, embed_size]
    lengths: batch of size [batch_size]
    indices: which token index in each batch example should be output
             shape: [batch_size] or [batch_size, 1]
    num_layers: number of stacked layers in the bi-RNN
    cell_type: type of RNN cell to use (e.g., LSTM, GRU)
    cell_size: cell's output size
    initial_state_fwd: initial state for forward direction
    initial_state_bwd: initial state for backward direction

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

  if scope is not None:
    scope_name = scope
  else:
    scope_name = "lbirnn"
  with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as varscope:
    #print("_lbirnn_helper scope={}".format(varscope))
    # reverse each batch example up through its length, maintaining right-padding
    inputs_rev = tf.reverse_sequence(inputs, lengths, batch_axis=0, seq_axis=1)

    cells_fwd = get_multi_cell(cell_type, cell_size, num_layers)
    cells_bwd = get_multi_cell(cell_type, cell_size, num_layers)

    if is_training and ("output_keep_prob" in kwargs) and (kwargs["output_keep_prob"] < 1.0):
      print("is_training={} --> using dropout in lbirnn (scope={})".format(is_training, scope_name))
      cells_fwd = tf.contrib.rnn.DropoutWrapper(cell=cells_fwd,
                                                output_keep_prob=kwargs["output_keep_prob"])
      cells_bwd = tf.contrib.rnn.DropoutWrapper(cell=cells_bwd,
                                                output_keep_prob=kwargs["output_keep_prob"])
    else:
      print("not using dropout in lbirnn (is_training={}, scope={})".format(is_training, scope_name))

    if "attention" in kwargs and kwargs["attention"] == True:
      if "attn_length" in kwargs:
        attn_length = kwargs["attn_length"]
      else:
        attn_length = 10
      cells_fwd = tf.contrib.rnn.AttentionCellWrapper(cells_fwd, attn_length=attn_length)
      cells_bwd = tf.contrib.rnn.AttentionCellWrapper(cells_bwd, attn_length=attn_length)


    batch_size = tf.shape(inputs)[0]
    if initial_state_fwd is None:
      initial_state_fwd = cells_fwd.zero_state(batch_size,
                                               tf.float32)
    else:
      # replace None values with zero states
      initial_state_fwd = list(initial_state_fwd)
      for i, c in enumerate(initial_state_fwd):
        if c is None:
          initial_state_fwd[i] = cell_type(cell_size).zero_state(batch_size,
                                                                 tf.float32)
      initial_state_fwd = tuple(initial_state_fwd)

    if initial_state_bwd is None:
      initial_state_bwd = cells_bwd.zero_state(batch_size,
                                               tf.float32)
    else:
      # replace None values with zero states
      initial_state_bwd = list(initial_state_bwd)
      for i, c in enumerate(initial_state_bwd):
        if c is None:
          initial_state_bwd[i] = cell_type(cell_size).zero_state(batch_size,
                                                                 tf.float32)
      initial_state_bwd = tuple(initial_state_bwd)

    outputs_fwd, last_state_fwd = tf.nn.dynamic_rnn(cells_fwd,
                                                    inputs,
                                                    sequence_length=lengths,
                                                    initial_state=initial_state_fwd,
                                                    time_major=False,
                                                    scope="rnn_fwd")

    tmp, last_state_bwd = tf.nn.dynamic_rnn(cells_bwd,
                                            inputs_rev,
                                            sequence_length=lengths,
                                            initial_state=initial_state_bwd,
                                            time_major=False,
                                            scope="rnn_bwd")
    # reverse backward-pass outputs so they align with the forward-pass outputs
    outputs_bwd = tf.reverse_sequence(tmp, lengths, batch_axis=0, seq_axis=1)

    if indices is not None:
      # row index [[0], [1], ..., [N]]
      r = tf.range(batch_size)
      r = tf.cast(r, dtype=tf.int64)
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
      outputs_fwd = tf.gather_nd(outputs_fwd, idx)
      outputs_bwd = tf.gather_nd(outputs_bwd, idx)

    return (outputs_fwd, outputs_bwd), (last_state_fwd, last_state_bwd)


def lbirnn(inputs,
           lengths,
           is_training,
           indices=None,
           num_layers=2,
           cell_type=tf.contrib.rnn.LSTMCell,
           cell_size=64,
           initial_state_fwd=None,
           initial_state_bwd=None,
           **kwargs):
  with tf.variable_scope("single-stage-lbirnn", reuse=tf.AUTO_REUSE) as varscope:
    o, _ = _lbirnn_helper(inputs,
                          lengths,
                          is_training=is_training,
                          indices=indices,
                          num_layers=num_layers,
                          cell_type=cell_type,
                          cell_size=cell_size,
                          initial_state_fwd=initial_state_fwd,
                          initial_state_bwd=initial_state_bwd,
                          scope=varscope,
                          **kwargs)
    (outputs_fwd, outputs_bwd) = o
    outputs = tf.concat([outputs_fwd, outputs_bwd], axis=-1)
    return outputs


def serial_lbirnn(inputs,
                  lengths,
                  is_training,
                  indices=None,
                  num_layers=2,
                  cell_type=tf.contrib.rnn.GRUCell,
                  cell_size=64,
                  initial_state_fwd=None,
                  initial_state_bwd=None,
                  **kwargs):
  """Serial stacked linear chain bi-directional RNN

  If `indices` is specified for the last stage, the outputs of the tokens
  in the last stage as specified by `indices` will be returned.
  If `indices` is None for the last stage, the encodings for all tokens
  in the sequence are returned.

  Inputs
  _____
    All arguments denoted with (*) should be given as lists,
    one element per stage in the series. The specifications given
    below are for a single stage.

    inputs (*): Tensor of size [batch_size, batch_len, embed_size]
    lengths (*): Tensor of size [batch_size]
    indices: Tensor of which token index in each batch item should be output;
             shape: [batch_size] or [batch_size, 1]
    num_layers: number of stacked layers in the bi-RNN
    cell_type: type of RNN cell to use (e.g., LSTM, GRU)
    cell_size: cell's output size
    initial_state_fwd: initial state for forward direction, may be None
    initial_state_bwd: initial state for backward direction, may be None

  Outputs
  _______
  If the input word vectors have dimension D and the series has N stages:
  if `indices` is not None:
    the output is a Tensor of size [batch_size, cell_size]
  if `indices` is None:
    the output is a Tensor of size [batch_size, batch_len, cell_size]
  """

  lists = [inputs, lengths]
  it = iter(lists)
  num_stages = len(next(it))
  if not all(len(l) == num_stages for l in it):
    raise ValueError("all list arguments must have the same length")

  assert num_stages > 0, "must specify arguments for " \
                         "at least one stage of serial bi-RNN"

  fwd_ = initial_state_fwd
  bwd_ = initial_state_bwd

  prev_scope = None
  for i in xrange(num_stages):
    #with tf.variable_scope("serial_lbirnn", reuse=tf.AUTO_REUSE) as varscope:
    with tf.variable_scope("serial_lbirnn_{}".format(i)) as varscope:
      if prev_scope is not None:
        #print("Previous scope={}".format(prev_scope))
        prev_scope.reuse_variables()
      inputs_ = inputs[i]
      lengths_ = lengths[i]
      if i == num_stages - 1:
        # Use the user-specified indices on the last stage
        indices_ = indices
      else:
        indices_ = None

      print("calling _lbirnn_helper() from serial_lbirnn()")
      o, s = _lbirnn_helper(inputs_,
                            lengths_,
                            is_training=is_training,
                            indices=indices_,
                            num_layers=num_layers,
                            cell_type=cell_type,
                            cell_size=cell_size,
                            initial_state_fwd=fwd_,
                            initial_state_bwd=bwd_,
                            scope=varscope,
                            **kwargs)
      (outputs_fwd, outputs_bwd), (last_state_fwd, last_state_bwd) = o, s
      # Update arguments for next stage
      fwd_ = last_state_fwd
      bwd_ = last_state_bwd
      prev_scope = varscope

  outputs = tf.concat([outputs_fwd, outputs_bwd], axis=-1)

  return outputs


def _lbirnn_stock(inputs,
                  lengths,
                  is_training,
                  num_layers=2,
                  cell_type=tf.contrib.rnn.GRUCell,
                  cell_size=64,
                  initial_state_fwd=None,
                  initial_state_bwd=None,
                  scope=None,
                  **kwargs):

  scope_name = scope if scope is not None else "stock-lbirnn"
  with tf.variable_scope(scope_name) as varscope:
    cells_fwd = get_multi_cell(cell_type, cell_size, num_layers)
    cells_bwd = get_multi_cell(cell_type, cell_size, num_layers)

    if is_training and ("output_keep_prob" in kwargs) and (kwargs["output_keep_prob"] < 1.0):
      print("is_training={} --> using dropout in stock lbirnn (scope={})".format(is_training, scope_name))
      cells_fwd = tf.contrib.rnn.DropoutWrapper(cell=cells_fwd,
                                                output_keep_prob=kwargs["output_keep_prob"])
      cells_bwd = tf.contrib.rnn.DropoutWrapper(cell=cells_bwd,
                                                output_keep_prob=kwargs["output_keep_prob"])
    else:
      print("not using dropout in stock lbirnn (is_training={}, scope={})".format(is_training, scope_name))

    if "attention" in kwargs and kwargs["attention"] == True:
      if "attn_length" in kwargs:
        attn_length = kwargs["attn_length"]
      else:
        attn_length = 10
      cells_fwd = tf.contrib.rnn.AttentionCellWrapper(cells_fwd, attn_length=attn_length)
      cells_bwd = tf.contrib.rnn.AttentionCellWrapper(cells_bwd, attn_length=attn_length)

    outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cells_fwd,
                                                           cells_bwd,
                                                           inputs,
                                                           sequence_length=lengths,
                                                           initial_state_fw=initial_state_fwd,
                                                           initial_state_bw=initial_state_bwd,
                                                           dtype=tf.float32)

    return outputs, last_states

  
def serial_lbirnn_stock(inputs,
                        lengths,
                        is_training,
                        num_layers=2,
                        cell_type=tf.contrib.rnn.GRUCell,
                        cell_size=64,
                        initial_state_fwd=None,
                        initial_state_bwd=None,
                        **kwargs):

  lists = [inputs, lengths]
  it = iter(lists)
  num_stages = len(next(it))
  if not all(len(l) == num_stages for l in it):
    raise ValueError("all list arguments must have the same length")

  assert num_stages > 0, "must specify arguments for " \
                         "at least one stage of serial bi-RNN"

  with tf.variable_scope("stock-lbirnn-seq1") as varscope1:
    _, seq1_states = _lbirnn_stock(inputs[0],
                                   lengths[0],
                                   #is_training,
                                   is_training=is_training,
                                   num_layers=num_layers,
                                   cell_type=cell_type,
                                   cell_size=cell_size,
                                   initial_state_fwd=initial_state_fwd,
                                   initial_state_bwd=initial_state_bwd,
                                   scope=varscope1)

  with tf.variable_scope("stock-lbirnn-seq2") as varscope2:
    varscope1.reuse_variables()
    outputs, states = _lbirnn_stock(inputs[1],
                                    lengths[1],
                                    #is_training,
                                    is_training=is_training,
                                    num_layers=num_layers,
                                    cell_type=cell_type,
                                    cell_size=cell_size,
                                    initial_state_fwd=seq1_states[0],
                                    initial_state_bwd=seq1_states[1],
                                    scope=varscope2)

  # concatenate hx_fwd and hx_bwd of top layer
  if num_layers > 1:
    output = tf.concat([states[0][-1][1], states[1][-1][1]], 1)
  else:
    output = tf.concat([states[0][1], states[1][1]], 1)

  return output


@registry.register_hparams
def RUDER_NAACL18_HPARAMS():
  hp = tf.contrib.training.HParams(
    cell_type='lstm',
    cell_size=256,
    num_layers=1,
    keep_prob=0.5
  )
  return hp


@registry.register_encoder
def ruder_encoder(inputs, lengths, is_training, hp=None):
  assert type(inputs) is list
  assert type(lengths) is list
  assert len(inputs) == len(lengths)
  assert len(inputs) == 2
  assert hp is not None

  num_input_dim = len(inputs[0].get_shape().as_list())
  assert num_input_dim == 3  # BATCH X TIME X EMBED
  num_length_dim = len(lengths[0].get_shape().as_list())
  assert num_length_dim == 1
  
  if hp.cell_type == 'gru':
    cell_type = tf.contrib.rnn.GRUCell
  elif hp.cell_type == 'lstm':
    cell_type = tf.contrib.rnn.LSTMCell
  else:
    raise ValueError(hp.cell_type)
  
  keep_prob = hp.keep_prob if is_training else 1.0

  code = serial_lbirnn(inputs,
                       lengths,
                       is_training=is_training,
                       num_layers=hp.num_layers,
                       cell_type=cell_type,
                       cell_size=hp.cell_size)

  assert len(code.get_shape().as_list()) == 2
  return code

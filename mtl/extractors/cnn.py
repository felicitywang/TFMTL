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

from mtl.util.reducers import reduce_max_over_time
from mtl.util.common import validate_extractor_inputs


def _conv_and_pool(inputs,
                   lengths=None,
                   num_filter=128,
                   max_width=7,
                   activation_fn=tf.nn.relu,
                   reducer=reduce_max_over_time):
  """Processes inputs using 1D convolutions of size [2, max_width] on
  the input followed by temporal pooling.

  Inputs
  ------
    inputs: batch of size [batch_size, batch_Len, embed_size]
    lengths: batch of size [batch_size] **ignored in this function**
    num_filter: number of filters for each width
    max_width: maximum filter width
    activation_fn: non-linearity to apply after convolutions. Can be None.
    reducer: pooling operation to apply to each convolved output

  Outputs
  -------
    If K different width filters are applied, the output is a Tensor of size
    [batch_size, num_filter * K].
  """
  filter_sizes = []
  for i in xrange(2, max_width + 1):
    filter_sizes.append((i + 1, num_filter))

  filters = []
  for width, num_filter in filter_sizes:
    conv_i = tf.layers.conv1d(
      inputs,
      num_filter,  # dimensionality of output space (num filters)
      width,  # length of the 1D convolutional window
      data_format='channels_last',  # (batch, time, embed_dim)
      strides=1,  # stride length of the convolution
      activation=activation_fn,
      padding='SAME',  # zero padding (left and right)
      name='conv_{}'.format(width))

    # Pooling
    # print("width={}, num_filter={}, inputs shape={}, conv_i shape={}".format(width, num_filter, inputs.get_shape().as_list(), conv_i.get_shape().as_list()))
    pool_i = reducer(conv_i, lengths=lengths, time_axis=1)

    # Append the filter
    filters.append(pool_i)

  return tf.concat(filters, 1)

def cnn_extractor(inputs,
                  lengths,
                  num_filter,
                  max_width,
                  activation_fn,
                  reducer):

  validate_extractor_inputs(inputs, lengths)

  num_stages = len(inputs)

  code = []
  prev_varscope = None
  for n_stage in xrange(num_stages):
    with tf.variable_scope("cnn-seq{}".format(n_stage)) as varscope:
      if prev_varscope is not None:
        prev_varscope.reuse_variables()
      if n_stage == 0:
        cond_inputs = inputs[0]
      else:
        # condition reading of seq_n on learned features of seq_n-1
        p = tf.expand_dims(p, axis=1)

        max_len = tf.reduce_max(lengths[n_stage], axis=0)  # get length of longest seq_n in batch)
        max_len = tf.reshape(max_len, [1])
        max_len = tf.cast(max_len, dtype=tf.int32)
        max_len = tf.concat([tf.constant([1]), max_len, tf.constant([1])], axis=0)  # [1, max_len, 1]

        p = tf.tile(p, max_len)  # tile over time dimension

        cond_inputs = tf.concat([inputs[n_stage], p], axis=2)  # condition via concatenation

        # mask out p for padded tokens
        mask = tf.sequence_mask(lengths[n_stage], dtype=tf.int32)
        mask = tf.expand_dims(mask, axis=2)
        mask = tf.cast(mask, dtype=tf.float32)
        cond_inputs = tf.multiply(cond_inputs, mask)

      p = _conv_and_pool(cond_inputs,
                         lengths=lengths[n_stage],
                         num_filter=num_filter,
                         max_width=max_width,
                         activation_fn=activation_fn,
                         reducer=reducer)

      prev_varscope = varscope


  outputs = p

  return outputs

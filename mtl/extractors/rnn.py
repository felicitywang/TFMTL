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


def rnn_and_pool(inputs,
                 lengths,
                 num_layers=2,
                 cell_type=tf.contrib.rnn.BasicLSTMCell,
                 cell_size=64,
                 initial_state=None,
                 reducer=reduce_max_over_time):

  cells = [cell_type(cell_size) for _ in xrange(num_layers)]
  cell = tf.contrib.rnn.MultiRNNCell(cells)

  if initial_state is None:
    batch_size = tf.shape(inputs)[0]
    initial_state = cell.zero_state(batch_size,
                                    tf.float32)

  # outputs has shape: <batch_size, batch_len, cell_size>
  outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                     inputs=inputs,
                                     sequence_length=lengths,
                                     time_major=False,
                                     initial_state=initial_state)

  # Pooling
  return reducer(outputs, lengths=lengths, time_axis=1)

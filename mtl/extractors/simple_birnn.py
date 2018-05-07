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

from six.moves import xrange

import numpy as np
import tensorflow as tf
import mtl.util.registry


@registry.register_hparams
def birnn_default():
  hp = tf.contrib.training.HParams(
    cell='lstm',
    size=256,
    depth=1,
    combine='concat',
    keep_prob=0.5
  )
  return hp


@registry.register_encoder
def birnn(inputs, length, hp=None, is_training=True):
  assert len(inputs.get_shape().as_list()) == 3
  batch_size = tf.shape(inputs)[0]
  keep_prob = hp.keep_prob if is_training else 1.0
  cell_fw = stacked_rnn_cell(hp.depth, hp.cell, hp.size, keep_prob, scope="fw")
  cell_bw = stacked_rnn_cell(hp.depth, hp.cell, hp.size, keep_prob, scope="bw")
  outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
    cell_fw,
    cell_bw,
    inputs,
    sequence_length=length,
    initial_state_fw=cell_fw.zero_state(batch_size, tf.float32),
    initial_state_bw=cell_bw.zero_state(batch_size, tf.float32),
    dtype=None,
    parallel_iterations=None,
    swap_memory=False,
    time_major=False,
    scope=None
  )
  fw, bw = output_states
  all_states = []
  assert type(fw) is tuple
  assert type(bw) is tuple
  fw = list(fw)
  bw = list(bw)
  for state in fw + bw:
    if 'lstm' in hp.cell:
      all_states += [state.h]
    else:
      all_states += state
  if hp.combine == 'concat':
    return tf.concat(all_states, axis=1)
  elif hp.combine == 'sum':
    return sum(all_states)
  else:
    raise ValueError(hp.combine)

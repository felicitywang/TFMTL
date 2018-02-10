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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops.init_ops import glorot_uniform_initializer
from tensorflow.python.ops.init_ops import zeros_initializer


def dense_layer(x, output_size, name, activation=None):
  """ Wrapper for building dense linear layers. """

  if activation == tf.nn.selu:
    init = tf.variance_scaling_initializer(scale=1.0, mode='fan_in')
  else:
    init = glorot_uniform_initializer()

  return tf.layers.dense(x, output_size, name=name,
                         kernel_initializer=init,
                         bias_initializer=zeros_initializer(),
                         activation=activation)

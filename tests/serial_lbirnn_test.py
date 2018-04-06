#! /usr/bin/env python
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

from mtl.extractors.lbirnn import serial_lbirnn


class SerialLBiRNNTests(tf.test.TestCase):
  def test_template(self):
    with self.test_session() as sess:

      i1 = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
      i1 = tf.reshape(i1, [2, 3, 1])
      l1 = tf.constant([3, 3])
      i2 = tf.constant([[7, 8], [9, 10]], dtype=tf.float32)
      i2 = tf.reshape(i2, [2, 2, 1])
      l2 = tf.constant([2, 2])
      i3 = tf.constant([[100], [101]], dtype=tf.float32)
      i3 = tf.reshape(i3, [2, 1, 1])
      l3 = tf.constant([1, 1])

      inputs = [i1, i2, i3]
      lengths = [l1, l2, l3]

      indices = None
      # indices = tf.constant([0,0])

      num_layers = 2
      # num_layers = 1

      cell_type = tf.contrib.rnn.GRUCell
      # cell_type = tf.contrib.rnn.BasicLSTMCell

      cell_size = 32
      initial_state_fwd = None
      initial_state_bwd = None
      outputs = serial_lbirnn(inputs,
                              lengths,
                              indices,
                              num_layers,
                              cell_type,
                              cell_size,
                              initial_state_fwd,
                              initial_state_bwd)

      all_variables = tf.global_variables()
      trainable_variables = tf.trainable_variables()

      init_ops = [tf.global_variables_initializer(),
                  tf.local_variables_initializer()]
      sess.run(init_ops)

      all_var, train_var, outputs_ = sess.run([all_variables,
                                               trainable_variables,
                                               outputs])

      print('All variables created...')
      for var in all_variables:
        print(var)

      print('Trainable variables created...')
      for var in trainable_variables:
        print(var)

      print(outputs_)
      print(outputs_.shape)


if __name__ == '__main__':
  tf.test.main()

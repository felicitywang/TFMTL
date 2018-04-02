#! /usr/bin/env python

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

import numpy as np
import tensorflow as tf

from mtl.util.reducers import reduce_avg_over_time as avg_over_time
from mtl.util.reducers import reduce_max_over_time as max_over_time
from mtl.util.reducers import reduce_min_over_time as min_over_time
from mtl.util.reducers import reduce_var_over_time as var_over_time


class ReducerTests(tf.test.TestCase):
  def test_min(self):
    X = [[1, 2, 1], [1, 2, 3]]
    x = tf.constant(X)
    op = min_over_time(x)
    with self.test_session() as sess:
      val = sess.run(op)
      self.assertEqual(val[0], min(X[0]))
      self.assertEqual(val[1], min(X[1]))

  def test_max(self):
    X = [[1, 2, 1], [2, 2, 3]]
    x = tf.constant(X)
    x_max = max_over_time(x)
    with self.test_session() as sess:
      x_max_val = sess.run(x_max)
      self.assertEqual(x_max_val[0], max(X[0]))
      self.assertEqual(x_max_val[1], max(X[1]))

  def test_avg(self):
    X = [[1., 2., 1.], [2., 2., 3.]]
    x = tf.constant(X)
    x_avg = avg_over_time(x)
    with self.test_session() as sess:
      x_avg_val = sess.run(x_avg)
      self.assertAlmostEqual(x_avg_val[0], sum(X[0]) / len(X[0]), places=4)
      self.assertAlmostEqual(x_avg_val[1], sum(X[1]) / len(X[1]), places=4)

  def test_avg_padding(self):
    X = [[1., 2., 1.], [2., 3., 0.]]
    L = [3., 2.]
    x = tf.constant(X)
    l = tf.constant(L)
    x_avg = avg_over_time(x, lengths=l)
    with self.test_session() as sess:
      x_avg_val = sess.run(x_avg)
      self.assertAlmostEqual(x_avg_val[0], sum(X[0]) / len(X[0]))
      self.assertAlmostEqual(x_avg_val[1], sum(X[1]) / (len(X[1]) - 1))

  def test_var(self):
    X = [[1., 2., 1.], [2., 2., 3.]]
    x = tf.constant(X)
    x_var = var_over_time(x)
    with self.test_session() as sess:
      val = sess.run(x_var)
      self.assertEqual(len(val.shape), 1)
      v1 = np.var(X[0])
      v2 = np.var(X[1])
      self.assertAlmostEqual(val[0], v1)
      self.assertAlmostEqual(val[1], v2)

  def test_var_padding(self):
    X = [[1., 2., 1.], [2., 2., 0.]]
    L = [3., 2.]
    x = tf.constant(X)
    l = tf.constant(L)
    x_var = var_over_time(x, lengths=l)
    with self.test_session() as sess:
      val = sess.run(x_var)
      self.assertEqual(len(val.shape), 1)
      v1 = float(np.var(X[0]))
      v2 = float(np.var(X[1][:-1]))
      self.assertAlmostEqual(val[0], v1)
      self.assertAlmostEqual(val[1], v2)


if __name__ == "__main__":
  tf.test.main()

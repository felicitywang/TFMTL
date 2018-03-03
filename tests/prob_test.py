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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from operator import mul

from mtl.mlvae.prob import normalize_logits
from mtl.mlvae.prob import marginal_log_prob
from mtl.mlvae.prob import conditional_log_prob


def slow_marginal(ln_joint_prob, target_dim):
  joint_prob = np.exp(ln_joint_prob)
  shape = joint_prob.shape
  target_dim_size = shape[target_dim + 1]
  batch_size = shape[0]
  assert np.sum(joint_prob) < float(batch_size) + 1.0e3
  assert np.sum(joint_prob) > float(batch_size) - 1.0e3
  ret = np.zeros([batch_size, target_dim_size])
  for i in xrange(batch_size):
    for j in xrange(target_dim_size):
      reduce_axis = [slice(None)] * len(joint_prob.shape)
      reduce_axis[0] = i
      reduce_axis[target_dim + 1] = j
      sum_slice = joint_prob[tuple(reduce_axis)]
      ret[i][j] = np.sum(joint_prob[tuple(reduce_axis)])
  assert np.sum(ret) < float(batch_size) + 1.0e3
  assert np.sum(ret) > float(batch_size) - 1.0e3
  return ret


def slow_conditional(ln_joint_prob, target_dim, cond_dim):
  p_cond = slow_marginal(ln_joint_prob, cond_dim)
  joint_prob = np.exp(ln_joint_prob)
  shape = joint_prob.shape
  target_dim_size = shape[target_dim + 1]
  cond_dim_size = shape[cond_dim + 1]
  batch_size = shape[0]
  ret = np.zeros([batch_size, cond_dim_size, target_dim_size])
  for i in xrange(batch_size):
    for j in xrange(cond_dim_size):
      for k in xrange(target_dim_size):
        reduce_axis = [slice(None)] * len(joint_prob.shape)
        reduce_axis[0] = i
        reduce_axis[cond_dim + 1] = j
        reduce_axis[target_dim + 1] = k
        ret[i][j][k] = np.sum(joint_prob[tuple(reduce_axis)]) / p_cond[i][j]
      assert np.sum(ret[i][j]) < 1.0 + 1.0e3
      assert np.sum(ret[i][j]) > 1.0 - 1.0e3
  return ret


class ProbTests(tf.test.TestCase):
  def test_unflat_and_normalize(self):
    batch_size = 2
    dims = [2, 3, 4]
    D = reduce(mul, dims)
    logits = tf.random_normal([batch_size, D])
    joint_dist = normalize_logits(logits, dims)
    total = tf.exp(tf.reduce_logsumexp(joint_dist))
    with self.test_session() as sess:
      self.assertAlmostEqual(sess.run(total), float(batch_size), places=3)

  def test_marginal_log_prob_2d(self):
    batch_size = 2
    dims = [2, 3]
    D = reduce(mul, dims)
    logits = tf.random_normal([batch_size, D])
    ln_joint = normalize_logits(logits, dims)
    px = tf.exp(marginal_log_prob(ln_joint, 0))
    py = tf.exp(marginal_log_prob(ln_joint, 1))
    with self.test_session() as sess:
      ln_joint_val, px_val, py_val = sess.run([ln_joint, px, py])
      self.assertEqual(len(ln_joint_val.shape), 3)
      self.assertEqual(ln_joint_val.shape[0], batch_size)
      self.assertEqual(ln_joint_val.shape[1], dims[0])
      self.assertEqual(ln_joint_val.shape[2], dims[1])

      true_px = slow_marginal(ln_joint_val, 0)
      true_py = slow_marginal(ln_joint_val, 1)

      self.assertEqual(px_val.shape[0], batch_size)
      self.assertEqual(px_val.shape[1], dims[0])
      self.assertEqual(py_val.shape[0], batch_size)
      self.assertEqual(py_val.shape[1], dims[1])
      for i in xrange(batch_size):
        self.assertAlmostEqual(np.sum(px_val[i]), 1.0, places=3)
        self.assertAlmostEqual(np.sum(py_val[i]), 1.0, places=3)
        for j in xrange(dims[0]):
          self.assertAlmostEqual(true_px[i][j], px_val[i][j], places=3)
        for j in xrange(dims[1]):
          self.assertAlmostEqual(true_py[i][j], py_val[i][j], places=3)

  def test_marginal_log_prob_3d(self):
    batch_size = 2
    dims = [2, 3, 4]
    D = reduce(mul, dims)
    logits = tf.random_normal([batch_size, D])
    ln_joint = normalize_logits(logits, dims)
    px = tf.exp(marginal_log_prob(ln_joint, 0))
    py = tf.exp(marginal_log_prob(ln_joint, 1))
    pz = tf.exp(marginal_log_prob(ln_joint, 2))
    with self.test_session() as sess:
      ln_joint_val, px_val, py_val, pz_val = sess.run([ln_joint, px, py, pz])
      self.assertEqual(len(ln_joint_val.shape), 4)
      self.assertEqual(ln_joint_val.shape[0], batch_size)
      self.assertEqual(ln_joint_val.shape[1], dims[0])
      self.assertEqual(ln_joint_val.shape[2], dims[1])
      self.assertEqual(ln_joint_val.shape[3], dims[2])
      true_px = slow_marginal(ln_joint_val, 0)
      true_py = slow_marginal(ln_joint_val, 1)
      true_pz = slow_marginal(ln_joint_val, 2)
      self.assertEqual(px_val.shape[0], batch_size)
      self.assertEqual(px_val.shape[1], dims[0])
      self.assertEqual(py_val.shape[0], batch_size)
      self.assertEqual(py_val.shape[1], dims[1])
      self.assertEqual(pz_val.shape[0], batch_size)
      self.assertEqual(pz_val.shape[1], dims[2])
      for i in xrange(batch_size):
        self.assertAlmostEqual(np.sum(px_val[i]), 1.0, places=3)
        self.assertAlmostEqual(np.sum(py_val[i]), 1.0, places=3)
        self.assertAlmostEqual(np.sum(pz_val[i]), 1.0, places=3)
        for j in xrange(dims[0]):
          self.assertAlmostEqual(true_px[i][j], px_val[i][j], places=3)
        for j in xrange(dims[1]):
          self.assertAlmostEqual(true_py[i][j], py_val[i][j], places=3)
        for j in xrange(dims[2]):
          self.assertAlmostEqual(true_pz[i][j], pz_val[i][j], places=3)

  def test_conditional_log_prob_2d(self):
    batch_size = 2
    dims = [2, 3]
    D = reduce(mul, dims)
    logits = tf.random_normal([batch_size, D])
    ln_joint = normalize_logits(logits, dims)
    px_y = tf.exp(conditional_log_prob(ln_joint, 0, 1))
    py_x = tf.exp(conditional_log_prob(ln_joint, 1, 0))
    with self.test_session() as sess:
      ln_joint_val, px_y_val, py_x_val = sess.run([ln_joint, px_y, py_x])
      self.assertEqual(len(ln_joint_val.shape), 3)
      self.assertEqual(ln_joint_val.shape[0], batch_size)
      self.assertEqual(ln_joint_val.shape[1], dims[0])
      self.assertEqual(ln_joint_val.shape[2], dims[1])
      true_px_y = slow_conditional(ln_joint_val, 0, 1)
      true_py_x = slow_conditional(ln_joint_val, 1, 0)
      self.assertEqual(px_y_val.shape[0], batch_size)
      self.assertEqual(px_y_val.shape[1], dims[1])
      self.assertEqual(px_y_val.shape[2], dims[0])
      self.assertEqual(py_x_val.shape[0], batch_size)
      self.assertEqual(py_x_val.shape[1], dims[0])
      self.assertEqual(py_x_val.shape[2], dims[1])
      for i in xrange(batch_size):
        for j in xrange(dims[1]):
          for k in xrange(dims[0]):
            self.assertAlmostEqual(true_px_y[i][j][k], px_y_val[i][j][k],
                                   places=3)

      for i in xrange(batch_size):
        for j in xrange(dims[0]):
          for k in xrange(dims[1]):
            self.assertAlmostEqual(true_py_x[i][j][k], py_x_val[i][j][k],
                                   places=3)

  def test_conditional_log_prob_3d(self):
    batch_size = 2
    dims = [2, 3, 4]
    D = reduce(mul, dims)
    logits = tf.random_normal([batch_size, D])
    ln_joint = normalize_logits(logits, dims)
    px_y = tf.exp(conditional_log_prob(ln_joint, 0, 1))
    py_x = tf.exp(conditional_log_prob(ln_joint, 1, 0))
    pz_x = tf.exp(conditional_log_prob(ln_joint, 2, 0))
    with self.test_session() as sess:
      ln_joint_val, px_y_val, py_x_val, pz_x_val = sess.run([ln_joint, px_y,
                                                             py_x, pz_x])
      self.assertEqual(len(ln_joint_val.shape), 4)
      self.assertEqual(ln_joint_val.shape[0], batch_size)
      self.assertEqual(ln_joint_val.shape[1], dims[0])
      self.assertEqual(ln_joint_val.shape[2], dims[1])
      self.assertEqual(ln_joint_val.shape[3], dims[2])
      true_px_y = slow_conditional(ln_joint_val, 0, 1)
      true_py_x = slow_conditional(ln_joint_val, 1, 0)
      true_pz_x = slow_conditional(ln_joint_val, 2, 0)
      self.assertEqual(px_y_val.shape[0], batch_size)
      self.assertEqual(px_y_val.shape[1], dims[1])
      self.assertEqual(px_y_val.shape[2], dims[0])
      self.assertEqual(py_x_val.shape[0], batch_size)
      self.assertEqual(py_x_val.shape[1], dims[0])
      self.assertEqual(py_x_val.shape[2], dims[1])
      self.assertEqual(pz_x_val.shape[0], batch_size)
      self.assertEqual(pz_x_val.shape[1], dims[0])
      self.assertEqual(pz_x_val.shape[2], dims[2])
      for i in xrange(batch_size):
        for j in xrange(dims[1]):
          for k in xrange(dims[0]):
            self.assertAlmostEqual(true_px_y[i][j][k], px_y_val[i][j][k],
                                   places=3)

      for i in xrange(batch_size):
        for j in xrange(dims[0]):
          for k in xrange(dims[1]):
            self.assertAlmostEqual(true_py_x[i][j][k], py_x_val[i][j][k],
                                   places=3)

      for i in xrange(batch_size):
        for j in xrange(dims[0]):
          for k in xrange(dims[2]):
            self.assertAlmostEqual(true_pz_x[i][j][k], pz_x_val[i][j][k],
                                   places=3)

if __name__ == "__main__":
  tf.test.main()

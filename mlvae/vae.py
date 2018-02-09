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


def mvn_diag_kl(p_loc=None, p_log_sigma=None, q_loc=None,
                q_log_sigma=None, pq=True):
  """Compute the analytic KL divergence between two multivariate
  Gaussians (p and q) with diagonal Sigma.

  """
  if pq:
    p_mu = tf.convert_to_tensor(p_loc)
    p_ls = tf.convert_to_tensor(p_log_sigma)
    q_mu = tf.convert_to_tensor(q_loc)
    q_ls = tf.convert_to_tensor(q_log_sigma)
  else:
    p_mu = tf.convert_to_tensor(q_loc)
    p_ls = tf.convert_to_tensor(q_log_sigma)
    q_mu = tf.convert_to_tensor(p_loc)
    q_ls = tf.convert_to_tensor(p_log_sigma)
  D = tf.to_float(tf.shape(p_mu)[-1])
  delta = p_mu - q_mu
  delta_sq = delta * delta
  a = tf.reduce_sum(q_ls, axis=1, keep_dims=True) - tf.reduce_sum(p_ls, axis=1, keep_dims=True)
  b = tf.reduce_sum(tf.exp(2.0 * (p_ls - q_ls)), axis=1, keep_dims=True)
  c = tf.reduce_sum(delta_sq * tf.exp(-2.0 * q_ls), axis=1, keep_dims=True)
  return a + 0.5 * (b - D + c)


def cross_entropy_with_logits(logits, targets):
  log_q = tf.nn.log_softmax(logits)
  return -tf.reduce_sum(targets * log_q, 1)


def log_normal(x, mu, var, axis=-1):
  return -0.5 * tf.reduce_sum(tf.log(2.0 * np.pi) + tf.log(var) +
                              tf.square(x - mu) / var, axis)


def gaussian_sample(mean, var):
  sample = tf.random_normal(tf.shape(mean), mean, tf.sqrt(var))
  sample.set_shape(mean.get_shape())
  return sample


def get_tau(hp, decay=False):
  if decay:
    global_step = tf.train.get_global_step()
    return tf.maximum(hp.tau_min,
                      tf.train.natural_exp_decay(
                        hp.tau0,
                        global_step,
                        hp.tau_decay_steps,
                        hp.tau_decay_rate,
                        staircase=False))
  else:
    return tf.constant(hp.tau0)

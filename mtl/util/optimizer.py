# Copyright 2017 Johns Hopkins University. All Rights Reserved.
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

import tensorflow as tf


class Optimizer(object):
  class H(object):
    optimizer = 'adam'
    max_grad_norm = 10
    decay_steps = 10000
    decay_rate = 1.0  # no decay
    min_lr = 0.00001
    lr = 0.0001
    use_nesterov = True
    momentum = 0.9
    rmsprop_decay = 0.9
    rmsprop_momentum = 0.9
    rmsprop_epsilon = 0.1
    adam_epsilon = 1e-6
    adam_beta1 = 0.85
    adam_beta2 = 0.997

  def __init__(self, global_step=None, config=H):
    self._config = config
    if global_step is None:
      self._global_step = tf.train.get_or_create_global_step()
    else:
      self._global_step = global_step
    if config.decay_rate < 1.0:
      self._lr = learning_rate = tf.maximum(
        config.min_lr,
        tf.train.exponential_decay(config.lr,
                                   self.global_step,
                                   config.decay_steps,
                                   config.decay_rate))
    else:
      self._lr = learning_rate = tf.get_variable(
        "learning_rate",
        shape=[],
        dtype=tf.float32,
        initializer=tf.constant_initializer(config.lr),
        trainable=False)
      self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
      self._lr_update = tf.assign(self._lr, self._new_lr)

    if config.optimizer == 'adam':
      self._opt = tf.train.AdamOptimizer(learning_rate,
                                         epsilon=config.adam_epsilon,
                                         beta1=config.adam_beta1,
                                         beta2=config.adam_beta2)
    elif config.optimizer == 'nadam':
      # Adam + Momentum
      self._opt = tf.contrib.opt.NadamOptimizer(learning_rate,
                                                beta1=config.adam_beta1,
                                                beta2=config.adam_beta2)
    elif config.optimizer == 'sgd':
      self._opt = tf.train.GradientDescentOptimizer(learning_rate)
    elif config.optimizer == 'adadelta':
      self._opt = tf.train.AdadeltaOptimizer(learning_rate)
    elif config.optimizer == 'momentum':
      self._opt = tf.train.MomentumOptimizer(
        learning_rate, config.momentum,
        use_nesterov=config.use_nesterov)
    elif config.optimizer == 'rmsprop':
      self._opt = tf.train.RMSPropOptimizer(
        learning_rate, config.rmsprop_decay,
        momentum=config.rmsprop_momentum,
        epsilon=config.rmsprop_epsilon)
    else:
      raise ValueError(
        'unrecognized optimizer: {}'.format(config.optimizer))

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def optimize(self, loss):
    return self.minimize(loss)

  def minimize(self, loss):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      self.config.max_grad_norm)
    self._train_op = self.opt.apply_gradients(zip(grads, tvars),
                                              global_step=self.global_step)
    return self.train_op

  @property
  def opt(self):
    return self._opt

  @property
  def train_op(self):
    return self._train_op

  @property
  def global_step(self):
    return self._global_step

  @property
  def lr(self):
    return self._lr

  @property
  def config(self):
    return self._config

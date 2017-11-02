# Copyright 2017 Johns Hopkins University. All Rights Reserved.
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

# Reference: https://arxiv.org/abs/1406.5298

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from collections import namedtuple

import numpy as np
import tensorflow as tf

from tensorflow.contrib.training import HParams

from tflm.models.vae_common import Inference
from tflm.models.vae_common import dense_layer
from tflm.models.vae_common import cross_entropy_with_logits
from tflm.models.vae_common import log_normal
from tflm.models.vae_common import gaussian_sample
from tflm.models.vae_common import get_tau


Categorical = tf.contrib.distributions.Categorical
ExpRelaxedOneHotCategorical = tf.contrib.distributions.ExpRelaxedOneHotCategorical
kl_divergence = tf.contrib.distributions.kl_divergence


def default_hparams():
  return HParams(embed_dim=256,
                 latent_dim=256,
                 tau0=0.25,
                 decay_tau=False,
                 inference=Inference.EXACT.value,
                 K=20,
                 dtype='float32')


LabeledLossResult = namedtuple('LabeledLossResult', 'loss nll kl_z')


class M2(object):
  def __init__(self, cond_vars=None, inputs=None, targets=None, decoders=None,
               hp=None, is_training=True):

    if hp is None:
      tf.logging.info("Using default hyper-parameters; none provided.")
      hp = default_hparams()

    if cond_vars is None:
      cond_vars = []
    elif type(cond_vars) is not list:
      cond_vars = [cond_vars]

    if type(inputs) is not list:
      inputs = [inputs]

    if type(targets) is not list:
      targets = [targets]

    if type(decoders) is not list:
      decoders = [decoders]

    self._cond_vars = cond_vars
    self._inputs = inputs
    self._targets = targets
    self._decoders = decoders

    # Keep track of batch size.
    self._batch_size = batch_size = tf.shape(inputs[0])[0]

    # Save hyper-parameters.
    self._hp = hp

    # p(z1 | y)
    def pz1_graph(y):
      zm = dense_layer(y, hp.latent_dim, 'zm')
      zv = dense_layer(y, hp.latent_dim, 'zv', tf.nn.softplus)
      return zm, zv

    # q(y | x)
    def qy_graph(inputs, output_size):
      if type(inputs) is list:
        x = tf.concat(inputs, axis=1)
      else:
        x = inputs

      x = dense_layer(x, hp.embed_dim, 'l1', tf.nn.elu)
      x = dense_layer(x, hp.embed_dim, 'l2', tf.nn.elu)
      x = dense_layer(x, output_size, 'logit', tf.nn.softplus)

      return x

    # q(z | x, y)
    def qz_graph(inputs):
      if type(inputs) is list:
        x = tf.concat(inputs, axis=1)
      else:
        x = inputs

      x = dense_layer(x, hp.embed_dim, 'l1', tf.nn.elu)
      x = dense_layer(x, hp.embed_dim, 'l2', tf.nn.elu)
      zm = dense_layer(x, hp.latent_dim, 'zm')
      zv = dense_layer(x, hp.latent_dim, 'zv', tf.nn.softplus)

      return zm, zv

    # Make sub-graph templates. Note that internal scopes and variable
    # names should not depend on any arguments that are not supplied
    # to make_template. In general you will get a ValueError telling
    # you that you are trying to reuse a variable that doesn't exist
    # if you make a mistake. Note that variables should be properly
    # re-used if the enclosing variable scope has reuse=True.

    # Generative networks
    pz1_template = tf.make_template('pz1', pz1_graph)

    # Inference networks
    qy_template = tf.make_template('qy', qy_graph, output_size=hp.K)
    qz_template = tf.make_template('qz', qz_graph)

    # Graph for q(y | x)
    if Inference(hp.inference) is Inference.EXACT:
      qy_logit = qy_template(inputs + cond_vars)
      qy = tf.nn.softmax(qy_logit)
      nent = -cross_entropy_with_logits(qy_logit, qy)
      y_ = tf.fill(tf.stack([batch_size, hp.K]), 0.0)
      losses = []
      for k in xrange(hp.K):
        y = tf.add(y_, tf.constant(np.eye(hp.K)[k], dtype=tf.float32))
        zm_prior, zv_prior = pz1_template(y)
        zm, zv = qz_template(inputs + cond_vars + [y])
        z = gaussian_sample(zm, zv)
        with tf.name_scope("loss_%d" % (k)):
          losses += [self.labeled_loss(y, z, zm, zv, zm_prior, zv_prior)]
    elif Inference(hp.inference) is Inference.SAMPLE:
      qy_logit = qy_template(inputs + cond_vars)
      qy = tf.nn.softmax(qy_logit)
      nent = -cross_entropy_with_logits(qy_logit, qy)
      tau = get_tau(hp, decay=hp.decay_tau)
      qy_concrete = ExpRelaxedOneHotCategorical(tau,
                                                logits=qy_logit,
                                                name='qy_concrete')
      log_y = qy_concrete.sample()
      y = tf.exp(log_y)
      py_logit = tf.log(tf.ones_like(qy_logit) * 1.0/float(hp.K))
      zm, zv = qz_template(inputs + cond_vars + [y])  # should this be e_y?
      z = gaussian_sample(zm, zv)
      zm_prior, zv_prior = pz1_template(y)
    else:
      raise ValueError('unrecognized inference mode: %s' % (hp.inference))

    if Inference(hp.inference) is Inference.EXACT:
      self._loss = tf.add_n([nent] + [qy[:, k] * losses[k].loss for k in
                                      xrange(hp.K)])
      self._kl_y = tf.constant(0.0)
      self._kl_z = tf.reduce_mean(tf.add_n([qy[:, k] *
                                            losses[k].kl_z for k in
                                            xrange(hp.K)]))
      self._nll = tf.reduce_mean(tf.add_n([qy[:, k] *
                                           losses[k].nll for k in
                                           xrange(hp.K)]))
      self._nent = tf.reduce_mean(nent)
      self._labels = tf.argmax(qy_logit, axis=1)
    elif Inference(hp.inference) is Inference.SAMPLE:
      self._nent = tf.reduce_mean(nent)
      self._labels = tf.argmax(qy_logit, axis=1)
      self._loss = self.approx_labeled_loss(y, qy_logit, py_logit, z,
                                            zm, zv, zm_prior,
                                            zv_prior)
    else:
      raise ValueError('unrecognized inference mode: %s' % (hp.inference))

    # Scalar summaries
    if is_training:
      tf.summary.scalar("NLL", self._nll)
      tf.summary.scalar("H_Y", self._nent)
      tf.summary.scalar("KL_y", self._kl_y)
      tf.summary.scalar("KL_z", self._kl_z)

  def get_var_grads(self):
    tvars = tf.trainable_variables()
    loss = tf.reduce_mean(self._loss)
    self._loss = loss
    grads = tf.gradients(loss, tvars)
    return (tvars, grads)

  def labeled_loss(self, y, z, zm, zv, zm_prior, zv_prior):
    nlls = []
    markov_blanket = tf.concat([z, y] + self._cond_vars, axis=1)
    for x, decoder in zip(self._targets, self._decoders):
      nlls += [decoder(x, markov_blanket)]
    nll = tf.add_n(nlls, name='accumulate_nll')
    lpy = np.log(1.0/self.hp.K)
    kl_z = log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
    loss = nll - lpy + kl_z
    return LabeledLossResult(loss, nll, kl_z)

  def approx_labeled_loss(self, y, qy_logit, py_logit, z, zm, zv,
                          zm_prior, zv_prior):
    nlls = []
    markov_blanket = tf.concat([z, y] + self._cond_vars, axis=1)
    for x, decoder in zip(self._targets, self._decoders):
      nlls += [decoder(x, markov_blanket)]
    nll = tf.add_n(nlls, name='accumulate_nll')
    qcat = Categorical(logits=qy_logit, name='qy_cat')
    pcat = Categorical(logits=py_logit, name='py_cat')
    kl_y = kl_divergence(qcat, pcat)
    kl_z = log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)

    # Diagnostics
    self._nll = tf.reduce_mean(nll)
    self._kl_y = tf.reduce_mean(kl_y)
    self._kl_z = tf.reduce_mean(kl_z)

    return nll + kl_y + kl_z

  @property
  def decoder(self):
    return self._decoder

  @property
  def labels(self):
    return self._labels

  @property
  def hp(self):
    return self._hp

  @property
  def loss(self):
    return self._loss

  @property
  def tau(self):
    return self._tau

  @property
  def nll(self):
    return self._nll

  @property
  def kl_y(self):
    return self._kl_y

  @property
  def kl_z(self):
    return self._kl_z

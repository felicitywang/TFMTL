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

import numpy as np
import tensorflow as tf

from tensorflow.contrib.training import HParams

from vae_common import Inference
from vae_common import dense_layer
from vae_common import cross_entropy_with_logits
from vae_common import log_normal
from vae_common import gaussian_sample
from vae_common import get_tau


Categorical = tf.contrib.distributions.Categorical
ExpRelaxedOneHotCategorical = tf.contrib.distributions.ExpRelaxedOneHotCategorical
kl_divergence = tf.contrib.distributions.kl_divergence


def default_hparams():
  return HParams(embed_dim=256,
                 latent_dim=256,
                 tau0=0.5,  # temperature
                 decay_tau=False,
                 alpha=0.1
                 # inference=Inference.EXACT.value,
                 # K=20,  # number of classes
                 dtype='float32')


class MultiLabel(object):
  def __init__(self, inputs=None, targets=None,
               feature_dict=None, observed_dict=None,
               class_sizes=None, decoder=None,
               hp=None, is_training=True):

    if hp is None:
      tf.logging.info("Using default hyper-parameters; none provided.")
      hp = default_hparams()

    if type(inputs) is not list:
      inputs = [inputs]

    if type(targets) is not list:
      targets = [targets]

    self._inputs = inputs
    self._targets = targets
    self._decoder = decoder

    # Keep track of batch size.
    self._batch_size = batch_size = tf.shape(inputs[0])[0]

    ## TODO: which indices in y are for observed labels, and which are for unobserved labels?
    # K_obs = ???
    # idx_obs = ???
    # K_unobs = ???
    # idx_unobs = ???

    # Save hyper-parameters.
    self._hp = hp

    # Distributions

    # Model A
    # p(x | y_o, y_u, z)
    # p(y_o | z)
    # p(y_u | z)
    # p(z)

    ######

    # Model B
    # p(x | y_o, y_u, z1)
    # p(y_o | z2)
    # p(y_u | z2)
    # p(z1)
    # p(z2)

    ######

    # Model C
    # p(x | y_o, y_u, z')
    # p(y_o | z)
    # p(y_u | z)
    # p(z' | z)
    # p(z)
    
    ######

    # Model D
    # p(x | y_o, y_u)
    # p(y_o | z)
    # p(y_u | z)
    # p(z)

    # ----
    
    # Inf y: A
    # q(y_i | x) (= q(y_o | x) * q(y_u | x))
    # def qyu_x_graph(inputs, output_size):
    #   assert(output_size == K_unobs)
    #   if type(inputs) is list:
    #     x = tf.concat(inputs, axis=1)
    #   else:
    #     x = inputs

    #   x = dense_layer(x, hp.embed_dim, 'l1', tf.nn.elu)
    #   x = dense_layer(x, hp.embed_dim, 'l2', tf.nn.elu)
    #   x = dense_layer(x, output_size, 'logit', tf.nn.softplus)

    #   return x

    # def qyo_x_graph(inputs, output_size):
    #   ## TODO: is this graph even necessary? the labels are observed, so we know for certain what they are
    #   assert(output_size == K_obs)
    #   if type(inputs) is list:
    #     x = tf.concat(inputs, axis=1)
    #   else:
    #     x = inputs

    #   x = dense_layer(x, hp.embed_dim, 'l1', tf.nn.elu)
    #   x = dense_layer(x, hp.embed_dim, 'l2', tf.nn.elu)
    #   x = dense_layer(x, output_size, 'logit', tf.nn.softplus)

    #   return x

    ######

    # Inf y: B
    # q(y_i | z) (= q(y_o | z) * q(y_u | z))
    # def qyu_z_graph(inputs, output_size):
    #   assert(output_size == K_unobs)
    #   if type(inputs) is list:
    #     x = tf.concat(inputs, axis=1)
    #   else:
    #     x = inputs

    #   x = dense_layer(x, hp.embed_dim, 'l1', tf.nn.elu)
    #   x = dense_layer(x, hp.embed_dim, 'l2', tf.nn.elu)
    #   x = dense_layer(x, output_size, 'logit', tf.nn.softplus)

    #   return x

    # def qyo_z_graph(inputs, output_size):
    #   ## TODO: is this graph even necessary? the labels are observed, so we know for certain what they are
    #   assert(output_size == K_obs)
    #   if type(inputs) is list:
    #     x = tf.concat(inputs, axis=1)
    #   else:
    #     x = inputs

    #   x = dense_layer(x, hp.embed_dim, 'l1', tf.nn.elu)
    #   x = dense_layer(x, hp.embed_dim, 'l2', tf.nn.elu)
    #   x = dense_layer(x, output_size, 'logit', tf.nn.softplus)

    #   return x

    ###### 

    # Inf y: C
    # q(y_i | x, z) (= q(y_o | x, z) * q(y_u | x, z))
    # def qyu_x_z_graph(inputs, output_size):
    #   assert(output_size == K_unobs)
    #   if type(inputs) is list:
    #     x = tf.concat(inputs, axis=1)
    #   else:
    #     x = inputs

    #   x = dense_layer(x, hp.embed_dim, 'l1', tf.nn.elu)
    #   x = dense_layer(x, hp.embed_dim, 'l2', tf.nn.elu)
    #   x = dense_layer(x, output_size, 'logit', tf.nn.softplus)

    #   return x

    # def qyo_x_z_graph(inputs, output_size):
    #   ## TODO: is this graph even necessary? the labels are observed, so we know for certain what they are
    #   assert(output_size == K_obs)
    #   if type(inputs) is list:
    #     x = tf.concat(inputs, axis=1)
    #   else:
    #     x = inputs

    #   x = dense_layer(x, hp.embed_dim, 'l1', tf.nn.elu)
    #   x = dense_layer(x, hp.embed_dim, 'l2', tf.nn.elu)
    #   x = dense_layer(x, output_size, 'logit', tf.nn.softplus)

    #   return x


    # ----

    # Inf z: A
    # q(z | y_o, y_u)
    # def qz_yo_yu_graph(inputs):
    #   if type(inputs) is list:
    #     x = tf.concat(inputs, axis=1)
    #   else:
    #     x = inputs

    #   x = dense_layer(x, hp.embed_dim, 'l1', tf.nn.elu)
    #   x = dense_layer(x, hp.embed_dim, 'l2', tf.nn.elu)
    #   zm = dense_layer(x, hp.latent_dim, 'zm')
    #   zv = dense_layer(x, hp.latent_dim, 'zv', tf.nn.softplus)

    #   return zm, zv

    # Inf z: B
    # q(z | x)
    # def qz_x_graph(inputs):
    #   if type(inputs) is list:
    #     x = tf.concat(inputs, axis=1)
    #   else:
    #     x = inputs

    #   x = dense_layer(x, hp.embed_dim, 'l1', tf.nn.elu)
    #   x = dense_layer(x, hp.embed_dim, 'l2', tf.nn.elu)
    #   zm = dense_layer(x, hp.latent_dim, 'zm')
    #   zv = dense_layer(x, hp.latent_dim, 'zv', tf.nn.softplus)

    #   return zm, zv

    # Inf z: C
    # q(z | x, y_o, y_u)
    # def qz_x_yo_yu_graph(inputs):
    #   if type(inputs) is list:
    #     x = tf.concat(inputs, axis=1)
    #   else:
    #     x = inputs

    #   x = dense_layer(x, hp.embed_dim, 'l1', tf.nn.elu)
    #   x = dense_layer(x, hp.embed_dim, 'l2', tf.nn.elu)
    #   zm = dense_layer(x, hp.latent_dim, 'zm')
    #   zv = dense_layer(x, hp.latent_dim, 'zv', tf.nn.softplus)

    #   return zm, zv
    
    ############################
    ##### REMOVE ###############
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

    #############################


    def MLP_gaussian_posterior(inputs):
      if type(inputs) is list:
        x = tf.concat(inputs, axis=1)
      else:
        x = inputs

      x = dense_layer(x, hp.embed_dim, 'l1', tf.nn.elu)
      x = dense_layer(x, hp.embed_dim, 'l2', tf.nn.elu)
      zm = dense_layer(x, hp.latent_dim, 'zm')
      zv = dense_layer(x, hp.latent_dim, 'zv', tf.nn.softplus)

      return zm, zv

    def MLP_unnormalized_categorical(inputs, output_size):
      if type(inputs) is list:
        x = tf.concat(inputs, axis=1)
      else:
        x = inputs

      x = dense_layer(x, hp.embed_dim, 'l1', tf.nn.elu)
      x = dense_layer(x, hp.embed_dim, 'l2', tf.nn.elu)
      x = dense_layer(x, output_size, 'logit', tf.nn.softplus)

      return x

    def MLP_ordinal(inputs):
      if type(inputs) is list:
        x = tf.concat(inputs, axis=1)
      else:
        x = inputs

      x = dense_layer(x, hp.embed_dim, 'l1', tf.nn.elu)
      x = dense_layer(x, hp.embed_dim, 'l2', tf.nn.elu)
      x = dense_layer(x, 1, 'val', None)

      return x

    ####################################
    
    # Make sub-graph templates. Note that internal scopes and variable
    # names should not depend on any arguments that are not supplied
    # to make_template. In general you will get a ValueError telling
    # you that you are trying to reuse a variable that doesn't exist
    # if you make a mistake. Note that variables should be properly
    # re-used if the enclosing variable scope has reuse=True.

    # Generative networks
    # pz1_template = tf.make_template('pz1', pz1_graph)

    # Inference networks
    qy_templates = dict()
    py_templates = dict()
    for k, v in feature_dict:
      qy_templates[k] = tf.make_template('qy_{}'.format(k), MLP_unnormalized_categorical, output_size=class_sizes[k])
      py_templates[k] = tf.make_template('py_{}'.format(k), MLP_unnormalized_categorical, output_size=class_sizes[k])

    # qy_template = tf.make_template('qy', qy_graph, output_size=hp.K)
    qz_template = tf.make_template('qz', MLP_gaussian_posterior)

    tau = get_tau(hp, decay=hp.decay_tau)

    # Sample (approximate inference)
    # y <- z <- x
    # y <- x <- z
    qy_logits = dict()
    py_logits = dict()
    sampled_observations = dict()
    predicted_log_ys = dict()

    # sample z from x
    zm, zv = qz_template(inputs)
    z = gaussian_sample(zm, zv)

    zm_prior = 0.0
    zv_prior = 1.0

    discriminative_labeled_loss = 0

    for k, v in observed_dict:
      if v is True:
        # this feature is observed, so don't sample
        sampled_observations[k] = feature_dict[k]

        # get predicted y[k] from inference network
        qy_logit = qy_templates[k](inputs + [z])
        qy_normalized = tf.nn.softmax(qy_logit)

        true_y = sampled_observations[k]

        true_one_hot_y = tf.one_hot(true_y, 1)

        # Here, rather than computing KL[qy_k || py_k], we compute 
        #   \alpha * \log q(true_y | ...)
        # which is the discriminative term from Kingma & Welling (2015)
        # where \alpha is a hyperparameter.

        # py_logit is unnormalized log distribution output of MLP
        py_logit = py_templates[k](z)

        qy_logits[k] = qy_logit
        py_logits[k] = py_logit

        # TODO: check TF documentation for xent functions and if they want logits or distributions
        discriminative_labeled_loss += alpha * xent(true_y, qy_logit/qy_normalized/???)

      else:
        # sample y from z, x
        qy_logit = qy_templates[k](inputs + [z])
        # qy_normalized = tf.nn.softmax(qy_logit)
        # nent += -cross_entropy_with_logits(qy_logit, qy)  # TODO: should this be += ?
        qy_concrete = ExpRelaxedOneHotCategorical(tau,
                                                  logits=qy_logit,
                                                  name='qy_{}_concrete'.format(k))
        predicted_log_y = qy_concrete.sample()  # relaxed one-hot vector in log space
        predicted_y = tf.exp(predicted_log_y)  # relaxed one-hot vector
        predicted_log_ys[k] = predicted_log_y

        py_logit = py_templates[k](z)

        qy_logits[k] = qy_logit
        py_logits[k] = py_logit

      # qy_logit = qy_template(inputs)
      # qy = tf.nn.softmax(qy_logit)
      # nent = -cross_entropy_with_logits(qy_logit, qy)
      # tau = get_tau(hp, decay=hp.decay_tau)
      # qy_concrete = ExpRelaxedOneHotCategorical(tau,
      #                                           logits=qy_logit,
      #                                           name='qy_concrete')
      # log_y = qy_concrete.sample()
      # y = tf.exp(log_y)
      # py_logit = tf.log(tf.ones_like(qy_logit) * 1.0/float(hp.K))
      # zm, zv = qz_template(inputs + [y])
      # z = gaussian_sample(zm, zv)
      # zm_prior, zv_prior = pz1_template(y)

    # self._nent = tf.reduce_mean(nent)
    # self._labels = tf.argmax(qy_logit, axis=1)
    # self._loss = self.approx_labeled_loss(y, qy_logit, py_logit, z,
    #                                       zm, zv, zm_prior,
    #                                       zv_prior)

    # self._nent = tf.reduce_mean(nent, axis=???)
    self._labels = dict()
    for k, v in qy_logits:
      sampled_observations[k] = tf.argmax(v, axis=1)
    self._labels = sampled_observations
    self._loss = self.approx_labeled_loss(sampled_observations, qy_logits, py_logits, z,
                                          zm, zv)

    # Scalar summaries
    if is_training:
      tf.summary.scalar("NLL", self._nll)
      # tf.summary.scalar("H_Y", self._nent)
      tf.summary.scalar("KL_y", self._kl_y)
      tf.summary.scalar("KL_z", self._kl_z)

  def get_var_grads(self):
    tvars = tf.trainable_variables()
    loss = tf.reduce_mean(self._loss)
    self._loss = loss
    grads = tf.gradients(loss, tvars)
    return (tvars, grads)

  def approx_labeled_loss(self, ys, qy_logits, py_logits, predicted_log_ys,
                          z, zm, zv,
                          zm_prior, zv_prior):
    # TODO: make sure we are averaging/adding losses correctly across labels and across batches

    nlls = []
    markov_blanket = tf.concat([z, y], axis=1)
    for x in self._targets:
      nlls += [self._decoder(x, markov_blanket)]
    nll = tf.add_n(nlls, name='accumulate_nll')

    kl_ys = dict()
    for k, v in feature_dict:
      qy_logit = qy_logits[k]
      py_logit = py_logits[k]

      if EXACT_EXPECTATION:
        qcat = Categorical(logits=qy_logit, name='qy_{}_cat'.format(k))
        pcat = Categorical(logits=py_logit, name='py_{}_cat'.format(k))
        kl_ys[k] = kl_divergence(qcat, pcat)
      else SAMPLE_EXPECTATION:
        #               ????        - Concrete(sampled_y | logits_qy)
        kl_ys[k] = log_p(sampled_y) - qy_concrete.log_p(predicted_log_ys[k])


    kl_z = log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)

    # Diagnostics
    self._nll = tf.reduce_mean(nll)
    self._kl_y = tf.reduce_mean(kl_ys)
    self._kl_z = tf.reduce_mean(kl_z)

    obs_label_loss = ???  # loss w.r.t. observed labels

    return nll + kl_y + kl_z + alpha*obs_label_loss

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

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

# [WIP]: TFLM dependencies
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
                 alpha=0.1,
                 expectation='exact',
                 num_z_samples=1,
                 num_y_samples=1,
                 # inference=Inference.EXACT.value,
                 # K=20,  # number of classes
                 dtype='float32')


class MultiLabel(object):
  def __init__(self, inputs=None, targets=None,
               feature_dict=None, observed_dict=None,
               class_sizes=None, decoder=None,
               hp=None, is_training=True):

    # inputs: representation of x
    # targets: integer-ID sequence of words in x

    # feature_dict: map from feature names to values (None if feature is unobserved)
    #   key: feature name
    #   values: Tensor of size <batch_size>, where each value in the Tensor is in range(0,...,|label_set|)
    # observed_dict: map from feature names to booleans (True if observed, False otherwise)
    # class_sizes: map from feature names to cardinality of label sets

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

    # Save hyper-parameters.
    self._hp = hp

    # Distributions
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
    py_templates = dict()
    for k, v in feature_dict:
      py_templates[k] = tf.make_template('py_{}'.format(k), MLP_unnormalized_categorical, output_size=class_sizes[k])

    # Inference networks
    qy_templates = dict()
    for k, v in feature_dict:
      qy_templates[k] = tf.make_template('qy_{}'.format(k), MLP_unnormalized_categorical, output_size=class_sizes[k])

    qz_template = tf.make_template('qz', MLP_gaussian_posterior)

    tau = get_tau(hp, decay=hp.decay_tau)

    # Sample (approximate inference)
    # Options:
    #   y <- z <- x
    #   y <- x <- z
    qy_logits = dict()
    py_logits = dict()
    observations = dict()
    predicted_log_ys = dict()

    # sample z from x
    zm, zv = qz_template(inputs)
    z = gaussian_sample(zm, zv)

    zm_prior = 0.0
    zv_prior = 1.0

    for k, v in observed_dict:
      if v is True:
        # this feature is observed, so don't sample
        observations[k] = feature_dict[k]

        # get predicted y[k] from inference network
        qy_logit = qy_templates[k](inputs + [z])
        # qy_normalized = tf.nn.softmax(qy_logit)

        true_y = sampled_observations[k]

        true_one_hot_y = tf.one_hot(true_y, 1)  # distribution over label set for label y: one-hot vector

        # Here, rather than computing KL[qy_k || py_k], we compute 
        #   \alpha * \log q(true_y | ...)
        # which is the discriminative term from Kingma & Welling (2015)
        # where \alpha is a hyperparameter.

        # py_logit is unnormalized log distribution output of MLP
        py_logit = py_templates[k](z)

        qy_logits[k] = qy_logit
        py_logits[k] = py_logit

        # TODO: check TF documentation for xent functions and if they want logits or distributions
        # discriminative_labeled_loss += alpha * xent(true_y, qy_logit/qy_normalized/???)

      else:
        # sample y from z, x
        qy_logit = qy_templates[k](inputs + [z])
        # qy_normalized = tf.nn.softmax(qy_logit)
        qy_concrete = ExpRelaxedOneHotCategorical(tau,
                                                  logits=qy_logit,
                                                  name='qy_{}_concrete'.format(k))

        predicted_log_y = qy_concrete.sample()  # relaxed one-hot vector in log space
        predicted_y = tf.exp(predicted_log_y)  # distribution over label set for label y: relaxed one-hot vector
        
        predicted_log_ys[k] = predicted_log_y

        py_logit = py_templates[k](z)

        qy_logits[k] = qy_logit
        py_logits[k] = py_logit

    self._labels = dict()
    for k, v in qy_logits:
      if k not in observations:
        observations[k] = tf.argmax(v, axis=1)
    self._labels = observations
    self._loss = self.approx_labeled_loss(observations, qy_logits, py_logits, z,
                                          zm, zv)

    # Scalar summaries
    if is_training:
      tf.summary.scalar("NLL", self._nll)
      tf.summary.scalar("KL_y", self._kl_y)
      tf.summary.scalar("KL_z", self._kl_z)

  def get_var_grads(self):
    # TODO: check if this is correct
    tvars = tf.trainable_variables()
    loss = tf.reduce_mean(self._loss)
    self._loss = loss
    grads = tf.gradients(loss, tvars)
    return (tvars, grads)

  def approx_labeled_loss(self, ys, qy_logits, py_logits, predicted_log_ys,
                          z, zm, zv,
                          zm_prior, zv_prior):
    # TODO: make sure we are averaging/adding losses correctly across labels and across batch

    zm, zv = qz_template(inputs)
    res = 0
    for _ in range(num_z_samples):
      z = gaussian_sample(zm, zv)
      res += log_normal(z, zm_prior, zv_prior)
    Eq_log_pz = res / num_z_samples

    # sum_Eq_log_py = 0
    # sum_Eq_log_qy = 0
    sum_kl_qp = 0
    sum_sampled_kl_qp = 0
    for k in feature_dict:
      res_p = 0
      res_q = 0
      res = 0
      if observed_dict[k] is True:
        # feature is observed (in all examples in this batch)
        zm, zv = qz_template(inputs)
        for _ in range(num_z_samples):
          z = gaussian_sample(zm, zv)
          log_py_k = tf.nn.log_softmax(py_templates[k](z))
          r = tf.range(0, batch_size, 1)
          r = tf.expand_dims(r, axis=0)
          idxs = feature_dict[k]
          idxs = tf.concat([tf.transpose(r), idxs], axis=1)
          vals = tf.gather_nd(log_py_k[i], idxs)  # get the feature_dict[k][i] element from log_py_k[i] (in batch mode)
          res_p += tf.reduce_sum(vals)
        Eq_log_py = res / num_z_samples
        Eq_log_qy = 0  # entropy of a degenerate point-mass (one-hot) probability distribution is 0
      else:
        # feature is unobserved (in all examples in this batch)
        if hp.expectation == 'exact':
          zm, zv = qz_template(inputs)
          for i in range(num_z_samples):
            z = gaussian_sample(zm, zv)
            qy_logits = qy_templates[k](inputs + [z])
            py_logits = py_templates[k](z)
            qcat = Categorical(logits=qy_logits, name='qy_{}_{}_cat'.format(k, i))
            pcat = Categorical(logits=py_logits, name='py_{}_{}_cat'.format(k, i))
            kl = kl_divergence(qcat, pcat)
            res += kl
          kl_qp = res / num_z_samples  # average KL divergence between q and p for feature k

            # z = gaussian_sample(zm, zv)
            # q_yk = tf.nn.softmax(qy_templates[k](inputs + [z]))
          #   log_py_k = tf.nn.log_softmax(py_templates[k](z))
          #   log_qy_k = tf.nn.log_softmax(qy_templates[k](inputs + [z]))
          #   dot_product_p = tf.reduce_sum(tf.multiply(q_yk, log_py_k), axis=1)
          #   dot_product_q = tf.reduce_sum(tf.multiply(q_yk, log_qy_k), axis=1)
          #   res_p += dot_product_p
          #   res_q += dot_product_q
          # Eq_log_py = res_p / num_z_samples
          # Eq_log_qy = res_q / num_z_samples
        elif hp.expectation == 'sample':
          zm, zv = qz_template(inputs)
          for i in range(num_z_samples):
            z = gaussian_sample(zm, zv)
            for _ in range(num_y_samples):
              qy_logits = qy_templates[k](inputs + [z])
              qy_concrete = ExpRelaxedOneHotCategorical(tau,
                                                        logits=qy_logits,
                                                        name='qy_{}_{}_concrete'.format(k, i))
              y_samples = tf.exp(qy_concrete.sample())  # each row is a continuous approximation to a categorical one-hot vector over label values
              res_q = qy_concrete.log_prob(y_samples)  # log q(y_samp)

              py_logits = py_templates[k](z)
              pcat = Categorical(logits=py_logits, name='py_samp_{}_{}_cat'.format(k, i))
              y_preds = tf.argmax(y_samples, axis=1)  # TODO: try annealing also
              # y_preds = tf.one_hot(y_preds, class_sizes[k])
              res_p = pcat.log_prob(y_preds)  # log p(y_samp)
              res += res_q - res_p  # == -(log_p - log_q) (a sampled value of KL(q || p))
          # Eq_log_py = res_p / (num_z_samples * num_y_samples)
          # Eq_log_qy = res_q / (num_z_samples * num_y_samples)
          sampled_kl_qp = res / (num_z_samples * num_y_samples)
        else:
          raise ValueError('unrecognized expectation mode: %s' % (hp.expectation))
      
      # accumulate totals over all features
      if hp.expectation == 'exact':
        sum_kl_qp += kl_qp  # total average KL divergence between q and p over all features
      elif hp.expectation == 'sample':
        # sum_Eq_log_py += Eq_log_py
        # sum_Eq_log_qy += Eq_log_qy
        sum_sampled_kl_qp += sampled_kl_qp
      else:
        raise ValueError('unrecognized expectation mode: %s' % (hp.expectation))

    res = 0
    zm, zv = qz_template(inputs)
    for _ in range(num_z_samples):
      z = gaussian_sample(zm, zv)
      for _ in range(num_y_samples):
        instantiation = []
        for k in feature_dict:
          inst = None
          if observed_dict[k] is False:
            # TODO: sample a value of y_k
            inst = ???
          else:
            inst = feature_dict[k]
          # TODO: put value of y in instantiation -- we need a feature ordering here
          pass
        markov_blanket = tf.concat([y], axis=1)  # parents of x in p model
        nll = self._decoder(self._targets, markov_blanket)  # reconstruction loss
        res += nll
    Eq_log_px = res / (num_z_samples * num_y_samples)

    res = 0
    zm, zv = qz_template(inputs)
    for _ in range(num_z_samples):
      z = gaussian_sample(zm, zv)
      res += log_normal(z, zm, zv)
    Eq_log_qz = res / num_z_samples


    # maximize the term in parentheses, which means minimize its negation
    #   this entire negated term is the cost (loss) to minimize
    if hp.expectation == 'exact':
      return -(Eq_log_pz - sum_kl_qp + Eq_log_px - Eq_log_qz)
    elif hp.expectation == 'sample':
      return -(Eq_log_pz - sum_sampled_kl_qp + Eq_log_px - Eq_log_qz)
    else:
      raise ValueError('unrecognized expectation mode: %s' % (hp.expectation))

    # kl_z = log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)  

    # z_loss = tf.zeros([batch_size])
    # # kl_ys = dict()
    # for k, v in feature_dict:
    #   qy_logit = qy_logits[k]
    #   py_logit = py_logits[k]

    #   if k in observed_dict:
    #     # add discriminative (alpha) term
    #     # TODO
    #     z_loss += alpha*obs_label_loss
    #     pass
    #   else:
    #     if EXPECTATION_MODE == 'exact':
    #       # Exact expectation
    #       qcat = Categorical(logits=qy_logit, name='qy_{}_cat'.format(k))
    #       pcat = Categorical(logits=py_logit, name='py_{}_cat'.format(k))

    #       z_loss += kl_divergence(qcat, pcat)
    #       # kl_ys[k] = kl_divergence(qcat, pcat)
    #     else:
    #       # TODO
    #       # Sample expectation
    #       #               ????        - Concrete(sampled_y | logits_qy)
    #       # kl_ys[k] = log_p(sampled_y) - qy_concrete.log_p(predicted_log_ys[k])
    #       z_loss += log_p(sampled_y) - qy_concrete.log_p(predicted_log_ys[k])

    # return nll + z_loss + kl_z

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

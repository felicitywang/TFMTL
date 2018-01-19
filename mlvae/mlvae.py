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

from encoders import cnn

Categorical = tf.contrib.distributions.Categorical
ExpRelaxedOneHotCategorical = tf.contrib.distributions.ExpRelaxedOneHotCategorical
kl_divergence = tf.contrib.distributions.kl_divergence

def default_hparams():
  return HParams(embed_dim=256,
                 latent_dim=256,
                 encode_dim=256,
                 word_embed_dim=256,
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
  def __init__(self,
               feature_dict=None,
               class_sizes=None,
               decoder=None,
               hp=None, is_training=True):

    # feature_dict: map from feature names to values (None if feature is unobserved)
    #   key: feature name
    #   values: Tensor of size <batch_size>, where each value in the Tensor is in range(0,...,|label_set|)
    # class_sizes: map from feature names to cardinality of label sets

    # Save hyper-parameters.
    if hp is None:
      tf.logging.info("Using default hyper-parameters; none provided.")
      hp = default_hparams()
    self._hp = hp

    self._decoder = decoder

    # Keep track of batch size.
    self._batch_size = batch_size = tf.shape(inputs[0])[0]

    ####################################
    
    # Make sub-graph templates. Note that internal scopes and variable
    # names should not depend on any arguments that are not supplied
    # to make_template. In general you will get a ValueError telling
    # you that you are trying to reuse a variable that doesn't exist
    # if you make a mistake. Note that variables should be properly
    # re-used if the enclosing variable scope has reuse=True.

    # Create templates for the parametrized parts of the computation
    # graph that are re-used in different places.
    self._encoder = tf.make_template(encoder_graph,
                                     encode_dim=hp.encode_dim)
    # Generative networks
    self._py_templates = dict()
    for k, v in feature_dict:
      self._py_templates[k] = tf.make_template('py_{}'.format(k), MLP_unnormalized_categorical, output_size=class_sizes[k])
    # Inference networks
    self._qy_templates = dict()
    for k, v in feature_dict:
      self._qy_templates[k] = tf.make_template('qy_{}'.format(k), MLP_unnormalized_categorical, output_size=class_sizes[k])
    self._qz_template = tf.make_template('qz', MLP_gaussian_posterior)

    self._tau = get_tau(hp, decay=hp.decay_tau)

    self._zm_prior = 0
    self._zv_prior = 1


    # TODO: assert that class_sizes[k] > max(feature_dict[k])


  # General helpers
  def listify(x):
    if type(x) is not list:
      return [x]
    else:
      return x

  def maybe_concat(x):
    if type(x) is list:
      return tf.concat(x, axis=1)
    else:
      return x

  # Encoding (feature extraction)
  def encode(self, targets):
    return self._encoder(targets)

  def encoder_graph(inputs, encode_dim, vocab_size):
    return cnn(inputs,
               input_size=vocab_size,
               embed_dim=word_embed_dim,
               encode_dim=encode_dim)

  # Distributions
  def preoutput_MLP(inputs, num_layers=2, activation=tf.nn.elu):
    # Returns output of last layer of N-layer dense MLP that can then be passed to an output layer
    x = maybe_concat(inputs)
    for i in range(num_layers):
      x = dense_layer(x, hp.embed_dim, 'l{}'.format(i+1), activation=activation)
    return x

  def MLP_gaussian_posterior(inputs):
    # Returns mean and variance parametrizing a (multivariate) Gaussian
    x = preoutput_MLP(inputs, num_layers=2, activation=tf.nn.elu)
    zm = dense_layer(x, hp.latent_dim, 'zm', activation=None)
    zv = dense_layer(x, hp.latent_dim, 'zv', tf.nn.softplus)  # variance must be positive

    return zm, zv

  def MLP_unnormalized_log_categorical(inputs, output_size):
    # Returns logits (unnormalized log probabilities)
    x = preoutput_MLP(inputs, num_layers=2, activation=tf.nn.elu)
    x = dense_layer(x, output_size, 'logit', activation=None)

    return x

  def MLP_ordinal(inputs):
    # Returns scalar output
    x = preoutput_MLP(inputs, num_layers=2, activation=tf.nn.elu)
    x = dense_layer(x, 1, 'val', activation=None)

    return x

  # Distribution + sampling helpers
  def sample_y(self, logits, name, argmax=False):
    # Returns an *approximately* one-hot vector representing a single sample of y (argmax is False)
    # OR
    # returns the index (label) with the highest probability (argmax is True)
    qy_concrete = ExpRelaxedOneHotCategorical(tau,
                                              logits=logits,
                                              name='qy_concrete_{}'.format(name))
    y_sample = tf.exp(qy_concrete.sample())
    if argmax:
      y_sample = tf.argmax(y_sample, axis=1)
    return y_sample

  def get_predictions(self, inputs, z, feature_name):
    # Returns most likely label given conditioning variables
    features = self.encode(inputs)
    logits = self._qy_templates[feature_name](features + [z])
    return tf.argmax(logits)

  def get_label_log_probability(self, inputs, z, label_idx, feature_name):
    # Returns the log probability (log q(y|x, z)) of a given label y
    features = self.encode(inputs)
    logits = self._qy_templates[feature_name](features + [z])
    log_dist = tf.nn.log_softmax(logits)
    r = tf.range(0, batch_size, 1)
    r = tf.expand_dims(r, axis=0)
    idx = feature_dict[feature_name]
    idx = tf.concat([tf.transpose(r), idx], axis=1)

    probs = tf.gather_nd(log_dist[i], idx)  # get the feature_dict[feature_name][i]'th element from log_dist[i] (in batch mode)
    return probs
    # return logits[label_idx]

  def get_var_grads(self):
    # TODO: check if this is correct
    tvars = tf.trainable_variables()
    loss = tf.reduce_mean(self._loss)
    self._loss = loss
    grads = tf.gradients(loss, tvars)
    return (tvars, grads)

  def get_loss(self,
               targets,
               labels,
               observed_dict,
               zm_prior,
               zv_prior,
               dataset_order):
    # TODO: make sure we are averaging/adding losses correctly across labels and across batch

    # features: representation of x
    # targets: integer-ID sequence of words in x
    # observed_dict: map from feature names to booleans (True if observed, False otherwise)

    # inputs = listify(inputs)
    targets = listify(targets)

    features = encode(targets)

    # TODO: rename `inputs` to `features` below

    batch_size = self._batch_size

    num_z_samples = hp.num_z_samples
    num_y_samples = hp.num_y_samples

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
        Eq_log_py = res_p / num_z_samples
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
              y_sample = tf.exp(qy_concrete.sample())  # each row is a continuous approximation to a categorical one-hot vector over label values
              res_q = qy_concrete.log_prob(y_sample)  # log q(y_samp)

              py_logits = py_templates[k](z)
              pcat = Categorical(logits=py_logits, name='py_samp_{}_{}_cat'.format(k, i))
              y_pred = tf.argmax(y_sample, axis=1)  # TODO: try annealing also
              # y_preds = tf.one_hot(y_preds, class_sizes[k])
              res_p = pcat.log_prob(y_pred)  # log p(y_samp)
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
        instantiation = dict()
        for k in feature_dict:
          inst = None
          if observed_dict[k] is False:
            # sample a value of y_k
            qy_logits = qy_templates[k](inputs + [z])
            qy_concrete = ExpRelaxedOneHotCategorical(tau,
                                                      logits=qy_logits,
                                                      name='qy_{}_{}_concrete'.format(k, i))
            y_sample = tf.exp(qy_concrete.sample())  # each row is a continuous approximation to a categorical one-hot vector over label values
            y_pred = tf.argmax(y_sample, axis=1)
            inst = y_pred
          else:
            inst = feature_dict[k]
          instantiation[k] = inst
        # instantiation dict to list with canonical ordering
        instantiation_list = [instantiation[k] for k in dataset_order]  # dataset_order: consistent ordering of tasks/datasets
        markov_blanket = tf.concat([instantiation_list], axis=1)  # parents of x in p model
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

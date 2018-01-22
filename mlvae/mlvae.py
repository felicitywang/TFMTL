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
                 dtype='float32')

# General helpers
def listify(x):
  # Convert inputs into a list if it is not a list already
  if type(x) is not list:
    return [x]
  else:
    return x

def maybe_concat(x):
  # Concatenate inputs if it is a list (of Tensors)
  # Provides a uniform way of conditioning on one variable vs multiple variables
  if type(x) is list:
    return tf.concat(x, axis=1)
  else:
    return x

def validate_labels(feature_dict, class_sizes):
  for k in feature_dict:
    assert class_sizes[k] > tf.reduce_max(feature_dict[k], axis=0), "Label for %s is out of range" % (k)

def encoder_graph(self, inputs, encode_dim, word_embed_dim, vocab_size):
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

class MultiLabel(object):
  def __init__(self,
               feature_set=None,
               class_sizes=None,
               dataset_order=None,
               decoder=None,
               hp=None):

    # feature_set: set containing names of features
    # class_sizes: map from feature names to cardinality of label sets

    # Save hyper-parameters.
    if hp is None:
      tf.logging.info("Using default hyper-parameters; none provided.")
      hp = default_hparams()
    self._hp = hp

    self._class_sizes = class_sizes

    self._dataset_order = dataset_order

    self._decoder = decoder

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
                                     encode_dim=hp.encode_dim,
                                     embed_dim=hp.word_embed_dim)
    # Generative networks
    self._py_templates = dict()
    for k in feature_set:
      self._py_templates[k] = tf.make_template('py_{}'.format(k), MLP_unnormalized_log_categorical, output_size=class_sizes[k])
    # Inference networks
    self._qy_templates = dict()
    for k in feature_set:
      self._qy_templates[k] = tf.make_template('qy_{}'.format(k), MLP_unnormalized_log_categorical, output_size=class_sizes[k])
    self._qz_template = tf.make_template('qz', MLP_gaussian_posterior)

    self._tau = get_tau(hp, decay=hp.decay_tau)

    self._zm_prior = 0.0
    self._zv_prior = 1.0

  # Encoding (feature extraction)
  def encode(self, targets):
    return self._encoder(targets)

  # Distribution + sampling helpers
  def sample_y(self, logits, name, argmax=False):
    # Returns an *approximately* one-hot vector representing a single sample of y (argmax is False)
    # OR
    # returns the index (label) with the highest probability (argmax is True)
    qy_concrete = ExpRelaxedOneHotCategorical(self._tau,
                                              logits=logits,
                                              name='qy_concrete_{}'.format(name))
    y_sample = tf.exp(qy_concrete.sample())
    if argmax:
      y_sample = tf.argmax(y_sample, axis=1)
    return y_sample

  def get_predictions(self, inputs, z, feature_name, features=None):
    # Returns most likely label given conditioning variables
    if features is None:
      features = self.encode(inputs)
    logits = self._qy_templates[feature_name](features + [z])
    return tf.argmax(logits, axis=1)

  def get_label_log_probability(self, feature_dict, features, z, feature_name, label_idx, distribution_type=None):
    # Returns the log probability (log p(y|z) or log q(y|x, z)) of a given label y
    # label_idx: Tensor of size <batch_size> that specifies which label index to query
    if distribution_type == 'p':
      logits = self._py_templates[feature_name](z)
    elif distribution_type == 'q':
      logits = self._qy_templates[feature_name](features + [z])
    else:
      raise ValueError('unrecognized distribution type: %s' % (distribution_type))
    log_dist = tf.nn.log_softmax(logits)

    r = tf.range(0, self._batch_size, 1)
    r = tf.expand_dims(r, axis=0)

    label_idx = tf.expand_dims(label_idx, axis=0)
    label_idx = tf.concat([tf.transpose(r), tf.transpose(label_idx)], axis=1)

    log_probs = tf.gather_nd(log_dist, label_idx)  # get the feature_dict[feature_name][i]'th element from log_dist[i] (in batch mode)
    return log_probs

  def get_label_instantiation(self, features, z, feature_dict, observed_dict):
    # Returns a dict (key=feature, value=label)
    #   that represents a full instantiation of all features
    # value = observed label for observed features
    # value = sampled label for unobserved features
    instantiation = dict()
    for k in feature_dict:
      if observed_dict[k] is True:
        instantiation[k] = feature_dict[k]
      else:
        # sample a value of y_k
        qy_logits = self._qy_templates[k](features + [z])
        instantiation[k] = self.sample_y(qy_logits, k, argmax=True)
    return instantiation

  def get_var_grads(self):
    # TODO: check if this is correct
    tvars = tf.trainable_variables()
    loss = tf.reduce_mean(self._loss)
    self._loss = loss
    grads = tf.gradients(loss, tvars)
    return (tvars, grads)

  def get_Eq_log_pz(self, zm, zv, zm_prior, zv_prior):
    res = 0
    for _ in range(hp.num_z_samples):
      z = gaussian_sample(zm, zv)
      res += log_normal(z, zm_prior, zv_prior)
    Eq_log_pz = res / hp.num_z_samples
    return Eq_log_pz

  def get_Eq_log_qz(self, zm, zv):
    res = 0
    for _ in range(hp.num_z_samples):
      z = gaussian_sample(zm, zv)
      res += log_normal(z, zm, zv)
    Eq_log_qz = res / hp.num_z_samples
    return Eq_log_qz

  def get_kl_qp(self, features, feature_name, feature_dict, observed_dict, zm, zv):
    kl_qp = 0
    if observed_dict[feature_name] is True:
      # feature is observed (in all examples in this batch)
      # Decouples into q and p terms (linearity of expectation)
      kl_qp += self.get_Eq_log_qy(feature_name, zm, zv) - self.get_Eq_log_py(features, feature_name, feature_dict, observed_dict, zm, zv)
    else:
      if hp.expectation == 'exact':
        # Does not easily decouple into q and p terms
        res = 0
        for i in range(hp.num_z_samples):
          z = gaussian_sample(zm, zv)
          
          qy_logits = self._qy_templates[feature_name](features + [z])
          qcat = Categorical(logits=qy_logits, name='qy_{}_{}_cat'.format(feature_name, i))
          
          py_logits = self._py_templates[feature_name](z)
          pcat = Categorical(logits=py_logits, name='py_{}_{}_cat'.format(feature_name, i))
          
          kl = kl_divergence(qcat, pcat)
          res += kl
        kl_qp += res / hp.num_z_samples  # average KL divergence between q and p for feature k
      elif hp.expectation == 'sample':
        # Decouples into q and p terms (linearity of expectation)
        kl_qp += self.get_Eq_log_qy(feature_name, zm, zv) - self.get_Eq_log_py(features, feature_name, feature_dict, observed_dict, zm, zv)
      else:
        raise ValueError('unrecognized expectation mode: %s' % (hp.expectation))

    return kl_qp

  def get_Eq_log_py(self, features, feature_name, feature_dict, observed_dict, zm, zv):
    Eq_log_py = 0
    if observed_dict[feature_name] is True:
      # feature is observed (in all examples in this batch)
      res_p = 0
      for _ in range(hp.num_z_samples):
        z = gaussian_sample(zm, zv)
        log_probs = self.get_label_log_probability(features, z, feature_name, feature_dict[feature_name], distribution_type='p')
        res_p += log_probs
      Eq_log_py = res_p / hp.num_z_samples
    else:
      if hp.expectation == 'exact':
        assert False  # this branch should not be reachable
      elif hp.expectation == 'sample':
        res = 0
        for i in range(hp.num_z_samples):
          z = gaussian_sample(zm, zv)
          qy_logits = self._qy_templates[feature_name](features + [z])
          qy_concrete = ExpRelaxedOneHotCategorical(self._tau,
                                                    logits=qy_logits,
                                                    name='qy_{}_{}_concrete'.format(feature_name, i))
          py_logits = self._py_templates[feature_name](z)
          pcat = Categorical(logits=py_logits, name='py_samp_{}_{}_cat'.format(feature_name, i))
          for _ in range(hp.num_y_samples):
            y_sample = tf.exp(qy_concrete.sample())  # each row is a continuous approximation to a categorical one-hot vector over label values
            y_pred = tf.argmax(y_sample, axis=1)  # TODO: try annealing also
            # y_preds = tf.one_hot(y_preds, class_sizes[k])
            res_p = pcat.log_prob(y_pred)  # log p(y_samp)
            res += res_p
        Eq_log_py = res / (hp.num_z_samples * hp.num_y_samples)
      else:
        raise ValueError('unrecognized expectation mode: %s' % (hp.expectation))
    return Eq_log_py

  def get_Eq_log_qy(self, features, feature_name, observed_dict, zm, zv):
    Eq_log_qy = 0
    if observed_dict[feature_name] is True:
      # feature is observed (in all examples in this batch)
      Eq_log_qy = 0  # entropy of a degenerate point-mass (one-hot) probability distribution is 0
    else:
      if hp.expectation == 'exact':
        assert False  # this branch should not be reachable
      elif hp.expectation == 'sample':
        res = 0
        for i in range(hp.num_z_samples):
          z = gaussian_sample(zm, zv)
          qy_logits = self._qy_templates[feature_name](features + [z])
          qy_concrete = ExpRelaxedOneHotCategorical(self._tau,
                                                    logits=qy_logits,
                                                    name='qy_{}_{}_concrete'.format(feature_name, i))
          for _ in range(hp.num_y_samples):
            y_sample = tf.exp(qy_concrete.sample())  # each row is a continuous approximation to a categorical one-hot vector over label values
            # TODO: do we need to create a qcat here and find the (log) probability of the y_pred according to qcat (like we do to calculate Eq_log_py)?
            res += qy_concrete.log_prob(y_sample)  # log q(y_sample)
        Eq_log_qy = res / (hp.num_z_samples * hp.num_y_samples)
      else:
        raise ValueError('unrecognized expectation mode: %s' % (hp.expectation))
    return Eq_log_qy

  def get_Eq_log_px(self, targets, features, feature_dict, observed_dict, zm, zv):
    res = 0
    for _ in range(hp.num_z_samples):
      z = gaussian_sample(zm, zv)
      for _ in range(hp.num_y_samples):
        instantiation = self.get_label_instantiation(features, z, feature_dict, observed_dict)
        # instantiation dict to list with canonical ordering
        instantiation_list = [instantiation[k] for k in self._dataset_order]  # dataset_order: consistent ordering of tasks/datasets
        markov_blanket = tf.concat([instantiation_list], axis=1)  # parents of x in p model
        nll = self._decoder(targets, markov_blanket)  # reconstruction loss
        res += nll
    Eq_log_px = res / (hp.num_z_samples * hp.num_y_samples)

  def get_disc_loss(self, features, feature_dict, feature_name, observed_dict, zm, zv):
    disc_loss = 0
    if observed_dict[feature_name] is True:
      z = gaussian_sample(zm, zv)
      disc_loss = self.get_label_log_probability(feature_dict, features, z, feature_name, feature_dict[feature_name], distribution_type='q')
    else:
      # TODO: does anything need to happen in this case?
      pass

    return disc_loss

  def get_loss(self,
               targets,
               feature_dict,
               features=None):
    # TODO: make sure we are averaging/adding losses correctly across labels and across batch

    # targets: integer-ID sequence of words in x
    # feature_dict: map from feature names to values (None if feature is unobserved)
    #   key: feature name
    #   values: Tensor of size <batch_size>, where each value in the Tensor is in range(0,...,|label_set|)

    validate_labels(feature_dict, self._class_sizes)

    targets = listify(targets)

    # Keep track of batch size.
    self._batch_size = tf.shape(targets[0])[0]

    # observed_dict: map from feature names to booleans (True if observed, False otherwise)
    observed_dict = {k : (v is not None) for k, v in feature_dict.items()}

    # features: representation of x
    if features is None:
      features = self.encode(targets)

    zm, zv = self._qz_template(features)

    Eq_log_pz = self.get_Eq_log_pz(zm, zv, self._zm_prior, self._zv_prior)
    total_kl_qp = 0
    for k in feature_dict:
      total_kl_qp += self.get_kl_qp(features, k, feature_dict, observed_dict, zm, zv)
    Eq_log_px = self.get_Eq_log_px(targets, features, feature_dict, observed_dict, zm, zv)
    Eq_log_qz = self.get_Eq_log_qz(zm, zv)

    total_disc_loss = 0
    for k in feature_dict:
      total_disc_loss += self.get_disc_loss(features, feature_dict, k, observed_dict, zm, zv)
    scaled_disc_loss = tf.reduce_mean(total_disc_loss, axis=0) * (hp.alpha * self._batch_size)


    # maximize the term in parentheses, which means minimize its negation
    #   this entire negated term is the cost (loss) to minimize
    self._loss = -(Eq_log_pz - total_kl_qp + Eq_log_px - Eq_log_qz - scaled_disc_loss)
    return self._loss

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

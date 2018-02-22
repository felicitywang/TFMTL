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

from six.moves import xrange

import tensorflow as tf

from tensorflow.contrib.training import HParams

from mtl.util.common import MLP_gaussian_posterior
from mtl.util.common import MLP_unnormalized_log_categorical

from mtl.mlvae.vae import log_normal
from mtl.mlvae.vae import gaussian_sample
from mtl.mlvae.vae import get_tau

logging = tf.logging

Categorical = tf.contrib.distributions.Categorical
ExpRelaxedOneHotCategorical = tf.contrib.distributions.ExpRelaxedOneHotCategorical
kl_divergence = tf.contrib.distributions.kl_divergence

def default_hparams():
  return HParams(embed_dim=256,
                 latent_dim=256,
                # encode_dim=128,
                 reuse_z=True,
                 word_embed_dim=256,
                 tau0=0.5,  # temperature
                 decay_tau=False,
                 alpha=0.1,
                 expectation='exact',
                 num_z_samples=1,
                 num_y_samples=1,
                 min_var=0.0001,
                 labels_key="label",
                 inputs_key="word_id",
                 targets_key="bow",
                 loss_type="discriminative",
                 loss_combination="even",
                 dtype='float32')



def validate_labels(feature_dict, class_sizes):
  for k in feature_dict:
    with tf.control_dependencies([
        tf.assert_greater(tf.cast(class_sizes[k], dtype=tf.int64),
                          tf.cast(tf.reduce_max(feature_dict[k]), dtype=tf.int64)
                        )]):
      pass


class MultiLabel(object):
  def __init__(self,
               class_sizes=None,
               dataset_order=None,
               encoders=None,
               decoders=None,
               encoder_features=None,
               decoder_features=None,  # x's that are being reconstructed; keys in a dictinary corresponding to input batches
               hp=None):

    # class_sizes: map from feature names to cardinality of label sets
    # dataset_order: list of features in some fixed order
    #   (concatenated order matters for decoding)
    # encoders: one per dataset
    # decoders: one per dataset

    assert class_sizes is not None
    assert dataset_order is not None
    assert encoders is not None
    assert decoders is not None

    # Save hyper-parameters.
    if hp is None:
      tf.logging.info("Using default hyper-parameters; none provided.")
      hp = default_hparams()
    self._hp = hp

    self._class_sizes = class_sizes

    self._dataset_order = dataset_order

    assert class_sizes.keys() == set(dataset_order)  # all feature names are present and consistent across data structures

    self._decoders = decoders

    self._encoders = encoders

    ####################################

    # Make sub-graph templates. Note that internal scopes and variable
    # names should not depend on any arguments that are not supplied
    # to make_template. In general you will get a ValueError telling
    # you that you are trying to reuse a variable that doesn't exist
    # if you make a mistake. Note that variables should be properly
    # re-used if the enclosing variable scope has reuse=True.

    # Create templates for the parametrized parts of the computation
    # graph that are re-used in different places.

    # Generative networks
    self._py_templates = dict()
    for k, v in class_sizes.items():
      self._py_templates[k] = tf.make_template('py_{}'.format(k), MLP_unnormalized_log_categorical, output_size=v, embed_dim=self._hp.embed_dim)
    # Inference networks
    self._qy_templates = dict()
    for k, v in class_sizes.items():
      self._qy_templates[k] = tf.make_template('qy_{}'.format(k), MLP_unnormalized_log_categorical, output_size=v, embed_dim=self._hp.embed_dim)
    self._qz_template = tf.make_template('qz', MLP_gaussian_posterior, embed_dim=self._hp.embed_dim, latent_dim=self._hp.latent_dim, min_var=self._hp.min_var)

    # NOTE: In general we will probably use a constant value for tau.
    self._tau = get_tau(self._hp, decay=self._hp.decay_tau)

    # We assume a standard N(0, 1) prior p(z) in some model
    # structures. Note that this is not necessary, and we may prefer
    # to use different priors, e.g. a parametrized
    # mixture-of-Gaussians.
    self._zm_prior = 0.0
    self._zv_prior = 1.0

  # Encoding (feature extraction)
  def encode(self, inputs, feature_name):
    return self._encoders[feature_name](inputs)

  def decode(self, x, z, feature_name):
    return self._decoders[feature_name](x, z)

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
      # NOTE: the argmax operation is not differentiable. This branch
      # should only be used for predictions at test time.
      y_sample = tf.argmax(y_sample, axis=1)
    return y_sample

  def get_predictions(self, batch, feature_name, features=None):
    # Returns most likely label given conditioning variables (only run this on eval data)
    inputs = batch[self._hp.inputs_key]

    if features is None:
      features = self.encode(inputs, feature_name)

    zm, zv = self._qz_template(features)
    z = gaussian_sample(zm, zv)

    logits = self._qy_templates[feature_name]([features, z])

    res = tf.argmax(logits, axis=1)
    res = tf.expand_dims(res, axis=1)

    return res

  def get_label_log_probability(self, feature_dict, features, z, feature_name, label_idx, distribution_type=None):
    # Returns the log probability (log p(y|z) or log q(y|x, z)) of a given label y
    # label_idx: Tensor of size <batch_size> that specifies which label index to query
    if distribution_type == 'p':
      logits = self._py_templates[feature_name](z)
    elif distribution_type == 'q':
      logits = self._qy_templates[feature_name]([features, z])
    else:
      raise ValueError('unrecognized distribution type: %s' % (distribution_type))
    log_dist = tf.nn.log_softmax(logits)
    log_dist = tf.reshape(log_dist, [self._batch_size, self._class_sizes[feature_name]])  # reshape to ignore first dimension

    # r = [[0], [1], ..., [batch_size-1]] <batch_size, 1>
    #   used as row indices for tf.gather_nd()
    r = tf.range(0, self._batch_size, delta=1)
    r = tf.expand_dims(r, axis=1)
    r = tf.cast(r, dtype=tf.int64)

    label_idx = tf.reshape(label_idx, [self._batch_size])
    label_idx = tf.expand_dims(label_idx, axis=1)  # <batch_size, 1>
    indices = tf.concat([r, label_idx], axis=1)  # indices used to query tf.gather_nd()

    log_probs = tf.gather_nd(log_dist, indices)  # get the feature_dict[feature_name][i]'th element from log_dist[i] (in batch mode)
    log_probs = tf.reshape(log_probs, [self._batch_size])
    return log_probs

  def get_label_instantiation(self, features, z, feature_dict, observed_dict):
    # Returns a dict (key=feature, value=label)
    #   that represents a full instantiation of all features
    # value = observed label for observed features
    # value = sampled label for unobserved features
    instantiation = dict()
    for k in feature_dict:
      if observed_dict[k] is True:
        instantiation[k] = tf.one_hot(feature_dict[k], self._class_sizes[k])  # one-hot
      else:
        # sample a value of y_k
        qy_logits = self._qy_templates[k]([features, z])
        instantiation[k] = self.sample_y(qy_logits, k, argmax=False)  # approx one-hot
    return instantiation

  def get_Eq_log_pz(self, zm, zv, zm_prior, zv_prior, z_samples=None):
    res = 0
    if z_samples is None:
      z_samples = [gaussian_sample(zm, zv) for _ in xrange(self._hp.num_z_samples)]
    for z in z_samples:
      res += log_normal(z, zm_prior, zv_prior)
    Eq_log_pz = res / len(z_samples)
    return Eq_log_pz

  def get_Eq_log_qz(self, zm, zv, z_samples=None):
    res = 0
    if z_samples is None:
      z_samples = [gaussian_sample(zm, zv) for _ in xrange(self._hp.num_z_samples)]
    for z in z_samples:
      res += log_normal(z, zm, zv)
    Eq_log_qz = res / len(z_samples)
    return Eq_log_qz

  def get_kl_qp(self, features, feature_name, feature_dict, observed_dict, zm, zv, z_samples=None):
    kl_qp = 0
    if observed_dict[feature_name] is True:
      # feature is observed (in all examples in this batch)
      # Decouples into q and p terms (linearity of expectation), so we calculate the q and p terms separately
      kl_qp += self.get_Eq_log_qy(feature_name, zm, zv, z_samples=z_samples) - self.get_Eq_log_py(features, feature_name, feature_dict, observed_dict, zm, zv, z_samples=z_samples)
    else:
      if self._hp.expectation == 'exact':
        # Does not easily decouple into q and p terms, so we calculate kl as a whole here
        res = 0
        if z_samples is None:
          z_samples = [gaussian_sample(zm, zv) for _ in xrange(self._hp.num_z_samples)]
        for z in z_samples:
          qy_logits = self._qy_templates[feature_name]([features, z])
          qcat = Categorical(logits=qy_logits, name='qy_{}_{}_cat'.format(feature_name, i))

          py_logits = self._py_templates[feature_name](z)
          pcat = Categorical(logits=py_logits, name='py_{}_{}_cat'.format(feature_name, i))

          kl = kl_divergence(qcat, pcat)
          res += kl
        kl_qp += res / len(z_samples)  # average KL divergence between q and p for feature k
      elif self._hp.expectation == 'sample':
        # Decouples into q and p terms (linearity of expectation), so we calculate the q and p terms separately
        kl_qp += self.get_Eq_log_qy(feature_name, zm, zv, z_samples=z_samples) - self.get_Eq_log_py(features, feature_name, feature_dict, observed_dict, zm, zv, z_samples=z_samples)
      else:
        raise ValueError('unrecognized expectation mode: %s' % (self._hp.expectation))

    return kl_qp

  def get_Eq_log_py(self, features, feature_name, feature_dict, observed_dict, zm, zv, z_samples=None):
    Eq_log_py = 0
    if observed_dict[feature_name] is True:
      # feature is observed (in all examples in this batch)
      res_p = 0
      if z_samples is None:
        z_samples = [gaussian_sample(zm, zv) for _ in xrange(self._hp.num_z_samples)]
      for z in z_samples:
        log_probs = self.get_label_log_probability(features, z, feature_name, feature_dict[feature_name], distribution_type='p')
        res_p += log_probs
      Eq_log_py = res_p / len(z_samples)
    else:
      if self._hp.expectation == 'exact':
        assert False, "this branch should not be reachable because kl for the exact case is calculated in get_kl_qp()"
      elif self._hp.expectation == 'sample':
        raise ValueError('sample expectation mode not supported: %s' % (self._hp.expectation))
      else:
        raise ValueError('unrecognized expectation mode: %s' % (self._hp.expectation))
    return Eq_log_py

  def get_Eq_log_qy(self, features, feature_name, observed_dict, zm, zv, z_samples=None):
    Eq_log_qy = 0
    if observed_dict[feature_name] is True:
      # feature is observed (in all examples in this batch)
      Eq_log_qy = 0  # entropy of a degenerate point-mass (one-hot) probability distribution is 0
    else:
      if self._hp.expectation == 'exact':
        assert False  # this branch should not be reachable because kl for the exact case is calculated in get_kl_qp()
      elif self._hp.expectation == 'sample':
        res = 0
        if z_samples is None:
          z_samples = [gaussian_sample(zm, zv) for _ in xrange(self._hp.num_z_samples)]
        for z in z_samples:
          qy_logits = self._qy_templates[feature_name]([features, z])
          qy_concrete = ExpRelaxedOneHotCategorical(self._tau,
                                                    logits=qy_logits,  # logits do *not* need to be manually exp-normalized for this distribution (the distribution automatically exp-normalizes)
                                                    name='qy_{}_{}_concrete'.format(feature_name, i))
          for _ in xrange(self._hp.num_y_samples):
            y_sample = tf.exp(qy_concrete.sample())  # each row is a continuous approximation to a categorical one-hot vector over label values
            # TODO: do we need to create a qcat here and find the (log) probability of the y_pred according to qcat (like we do to calculate Eq_log_py)?
            res += qy_concrete.log_prob(y_sample)  # log q(y_sample)
        Eq_log_qy = res / (len(z_samples) * self._hp.num_y_samples)
      else:
        raise ValueError('unrecognized expectation mode: %s' % (self._hp.expectation))
    return Eq_log_qy

  def get_Eq_log_px(self, targets, features, feature_dict, observed_dict, zm, zv, z_samples=None):
    res = 0
    if z_samples is None:
      z_samples = [gaussian_sample(zm, zv) for _ in xrange(self._hp.num_z_samples)]
    for z in z_samples:
      for _ in xrange(self._hp.num_y_samples):
        instantiation = self.get_label_instantiation(features, z, feature_dict, observed_dict)
        # instantiation dict to list with canonical ordering
        instantiation_list = [instantiation[k] for k in self._dataset_order]  # dataset_order: consistent ordering of tasks/datasets
        markov_blanket = tf.concat([instantiation_list], axis=1)  # parents of x in p model
        _feature_name = set(f for f in observed_dict if observed_dict[f] is True)
        assert len(_feature_name) == 1
        nll = self.decode(targets, markov_blanket, _feature_name)  # reconstruction loss
        res += nll
    Eq_log_px = res / (len(z_samples) * self._hp.num_y_samples)

  def get_disc_loss(self, features, feature_dict, feature_name, observed_dict, zm, zv, z_samples=None):
    disc_loss = 0
    if observed_dict[feature_name] is True:
      res = 0
      if z_samples is None:
        z_samples = [gaussian_sample(zm, zv) for _ in xrange(self._hp.num_z_samples)]
      for z in z_samples:
        # We want to maximize the log probability, which means minimize its negative
        res += -1.0 * self.get_label_log_probability(feature_dict, features, z, feature_name, feature_dict[feature_name], distribution_type='q')
      disc_loss = res / len(z_samples)
    else:
      #Discriminative loss not defined for unobserved examples
      pass

    return disc_loss

  def get_multi_task_loss(self, dataset_batches):
    # dataset_batches: map from dataset names to training batches (one batch per dataset)
    # we assume only one dataset's labels are observed; the rest are unobserved

    total_loss = 0
    for dataset_name, batch in dataset_batches.items():
      # We assume that the encoders and decoders always use the same fields/features (given by the keys in the batch accesses below)
      inputs = batch[self._hp.inputs_key]
      targets = batch[self._hp.targets_key]
      labels = batch[self._hp.labels_key]

      feature_dict = dict()
      for dataset_name_2 in dataset_batches:
        if dataset_name == dataset_name_2:
          feature_dict[dataset_name] = labels
        else:
          feature_dict[dataset_name_2] = None

      if self._hp.loss_combination == "even":
        total_loss += self.get_loss(feature_dict, inputs=inputs, targets=targets, loss_type=self._hp.loss_type, features=None)
      else:
        raise ValueError("unrecognized loss combination type: %s" % (self._hp.loss_combination))

    return total_loss

  def get_total_discriminative_loss(self, features, feature_dict, observed_dict, zm, zv, z_samples=None):
    total_disc_loss = 0
    for k in feature_dict:
      total_disc_loss += self.get_disc_loss(features, feature_dict, k, observed_dict, zm, zv, z_samples=z_samples)

    return total_disc_loss

  def get_total_generative_loss(self, features, feature_dict, observed_dict, targets, zm, zv, z_samples=None):
    # TODO(noa): this doesn't support computing KL[q(z) || p(z)]
    # analytically, since Eq_log_pz and Eq_log_qz are separate terms
    # in the loss below. Instead, we should have a KL_z method that
    # support *either* analytic or MCMC modes.

    Eq_log_pz = self.get_Eq_log_pz(zm, zv, self._zm_prior, self._zv_prior, z_samples=z_samples)
    total_kl_qp = 0
    for k in feature_dict:
      total_kl_qp += self.get_kl_qp(features, k, feature_dict, observed_dict, zm, zv, z_samples=z_samples)
    Eq_log_px = self.get_Eq_log_px(targets, features, feature_dict, observed_dict, zm, zv, z_samples=z_samples)
    Eq_log_qz = self.get_Eq_log_qz(zm, zv, z_samples=z_samples)

    # We want to maximize the term in parentheses,
    # which means minimize its negation
    # This entire negated term is the cost (loss) to minimize

    return -(Eq_log_pz - total_kl_qp + Eq_log_px - Eq_log_qz)

  def get_loss(self,
               feature_dict,
               inputs=None,
               targets=None,
               loss_type=None,
               features=None):
    # TODO: make sure we are averaging/adding losses correctly across labels and across batch

    # inputs: integer IDs of words in x
    #
    # targets: what the decoder is trying to reconstruct
    # NOTE: inputs and targets represent the same text. they are usually the same representation,
    #         although they may be different representations
    #         e.g., inputs: sequence of words
    #               targets: bag of words
    #
    # feature_dict: map from feature names to values (None if feature is unobserved)
    #   key: feature name
    #   values: Tensor of size <batch_size>, where each value in the Tensor is in range(0,...,|label_set|)
    #
    # features: representation of inputs, e.g., from a CNN (*NOT* names of labels or the values of labels)
    # reuse_z: whether to reuse the same samples of z throughout the computation (True: reuse, False: resample throughout)

    assert inputs is not None or features is not None
    assert targets is not None
    assert loss_type is not None

    validate_labels(feature_dict, self._class_sizes)

    # Keep track of batch length and size.
    # self._batch_size = tf.shape(targets[0])[0]
    self._batch_size = tf.shape(inputs)[0]

    # observed_dict: map from feature names to booleans (True if observed, False otherwise)
    observed_dict = {k : (v is not None) for k, v in feature_dict.items()}

    # features: representation of x
    if features is None:
      _feature_name = set(f for f in observed_dict if observed_dict[f] is True)
      assert len(_feature_name) == 1
      for _f in _feature_name:
        features = self.encode(inputs, _f)  # encode with encoder corresponding to observed feature

    zm, zv = self._qz_template(features)

    if self._hp.reuse_z is True:
      z_samples = [gaussian_sample(zm, zv) for _ in xrange(self._hp.num_z_samples)]
    else:
      z_samples = None

    if loss_type == 'discriminative':
      logging.info("MODE: discriminative loss")
      # Discriminative loss
      d_loss = self.get_total_discriminative_loss(features, feature_dict, observed_dict, zm, zv, z_samples=z_samples)
      assert len(d_loss.get_shape()) == 1, "expected dim(d_loss) == 1"
      loss = tf.reduce_mean(d_loss, axis=0)  # average across batch

    elif loss_type == 'gen+disc':
      logging.info("MODE: generative + discriminative loss")
      # Generative loss
      g_loss = self.get_total_generative_loss(features, feature_dict, observed_dict, targets, zm, zv, z_samples=z_samples)

      # Discriminative loss, scaled by a hyperparameter
      # See: https://arxiv.org/abs/1406.5298
      d_loss = self.get_total_discriminative_loss(features, feature_dict, observed_dict, zm, zv, z_samples=z_samples)
      scaled_disc_loss = tf.reduce_mean(d_loss, axis=0) * self._hp.alpha

      loss = g_loss + scaled_disc_loss

      # The loss at this point should be a vector of size batch_size,
      # therefore reduce_mean below is over the batch dimension.
      loss = tf.reduce_mean(loss)

    else:
      raise ValueError("unrecognized loss type: %s" % (loss_type))

    return loss

  @property
  def encoders(self):
    return self._encoders

  @property
  def decoders(self):
    return self._decoders

  @property
  def hp(self):
    return self._hp

  @property
  def tau(self):
    return self._tau

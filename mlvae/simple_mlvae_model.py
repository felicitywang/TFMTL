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

from mlvae.common import MLP_gaussian_posterior
from mlvae.common import MLP_unnormalized_log_categorical

from mlvae.vae import log_normal
from mlvae.vae import gaussian_sample
from mlvae.vae import get_tau

logging = tf.logging

Categorical = tf.contrib.distributions.Categorical
ExpRelaxedOneHotCategorical = tf.contrib.distributions.ExpRelaxedOneHotCategorical
kl_divergence = tf.contrib.distributions.kl_divergence


def default_hparams():
  return HParams(embed_dim=256,
                 latent_dim=256,
                 tau0=0.5,
                 decay_tau=False,
                 alpha=10.0,
                 expectation="exact",
                 labels_key="label",
                 inputs_key="inputs",
                 targets_key="targets",
                 loss_type="gd",
                 loss_reduce="even",
                 dtype='float32')


def generative_loss(decoder, x, markov_blanket, ys, qy_logits,
                    py_logits, z, zm, zv, zm_prior, zv_prior):
  nll_x = decoder(x, markov_blanket)
  nll_ys = []
  kl_ys = []
  for k in ys.keys():
    qy_logit = qy_logits[k]
    py_logit = py_logits[k]
    if qy_logit:
      qcat = Categorical(logits=qy_logit, name='qy_cat_{}'.format(k))
      pcat = Categorical(logits=py_logit, name='py_cat_{}'.format(k))
      kl_ys += [kl_divergence(qcat, pcat)]
    else:
      nll_ys += tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=py_logit,
        labels=ys[k])
  assert len(nll_ys) > 0
  assert len(kl_ys) > 0
  nll_y = tf.add_n(nll_ys)
  kl_y = tf.add_n(kl_ys)
  kl_z = log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
  return nll_x + nll_y + kl_y + kl_z


class SimpleMultiLabel(object):
  def __init__(self,
               class_sizes=None,
               encoders=None,
               decoders=None,
               hp=None):
    """
    class_sizes: map from feature names to cardinality of label sets
    dataset_order: *ordered* list of features
    encoders: one per label (i.e. dataset)
    decoders: one per label (i.e. dataset)
    """

    assert class_sizes is not None
    assert encoders.keys() == decoders.keys()

    self._encoders = encoders
    self._decoders = decoders

    if hp is None:
      tf.logging.info("Using default hyper-parameters; none provided.")
      hp = default_hparams()
    self._hp = hp

    self._class_sizes = class_sizes

    ########################### Generative Networks ###########################

    # p(y_1), ..., p(y_K)
    self._py_templates = dict()
    for k, v in class_sizes.items():
      self._py_templates[k] = tf.make_template(
        'py_{}'.format(k),
        MLP_unnormalized_log_categorical,
        output_size=v,
        embed_dim=hp.embed_dim)

    # p(z | y_1, ..., y_K)
    self._pz_template = tf.make_template('pz', MLP_gaussian_posterior,
                                         embed_dim=hp.embed_dim,
                                         latent_dim=hp.latent_dim)

    ########################### Inference Networks ############################

    # q(y_1 | x), ..., q(y_K | x)
    self._qy_templates = dict()
    for k, v in class_sizes.items():
      self._qy_templates[k] = tf.make_template(
        'qy_{}'.format(k),
        MLP_unnormalized_log_categorical,
        output_size=v,
        embed_dim=hp.embed_dim)

    # q(z | y_1, ..., y_K)
    self._qz_template = tf.make_template('qz', MLP_gaussian_posterior,
                                         embed_dim=hp.embed_dim,
                                         latent_dim=hp.latent_dim)


    # NOTE: In general we will probably use a constant value for tau.
    self._tau = get_tau(hp, decay=hp.decay_tau)

  def py_logits(self, k):
    return self._py_templates[k]()

  def qy_logits(self, k, features):
    return self._qy_templates[k](features)

  def pz_mean_var(self, ys):
    return self._pz_template(ys)

  def qz_mean_var(self, features, ys):
    return self._qz_template([features] + ys)

  def sample_y(self, logits, name):
    log_qy = ExpRelaxedOneHotCategorical(self._tau,
                                         logits=logits,
                                         name='log_qy_{}'.format(name))
    y = tf.exp(log_qy.sample())
    return y

  def get_predictions(self, inputs, feature_name):
    features = self.encode(inputs, feature_name)
    logits = self.qy_logits(feature_name, features)
    probs = tf.nn.softmax(logits)
    return tf.argmax(probs, axis=1)

  def get_total_discriminative_loss(self, features, labels):
    nll_ys = []
    for k, label in labels.items():
      if label is not None:
        nll_ys += [tf.sparse_softmax_cross_entropy_with_logits(
          labels=label,
          logits=self.qy_logits(features),
          name='d_loss_{}'.format(k))]
    assert len(nll_ys)
    return tf.add_n(nll_ys)

  def get_total_generative_loss(self, features, decoder, targets,
                                labels):
    ys = []
    qy_logits = {}
    py_logits = {}
    for k, label in labels.items():
      py_logits[k] = self.py_logits()
      if label is None:
        qy_logits[k] = self.qy_logits(features)
        ys += [self.sample_y(qy_logits[k])]
      else:
        qy_logits[k] = None
        ys += [tf.nn.one_hot(label, self._class_sizes[k])]
    zm, zv = self.qz_mean_var(features, ys)
    z = gaussian_sample(zm, zv)
    zm_prior, zv_prior = self.pz_mean_var(ys)
    markov_blanket = [z]
    return generative_loss(decoder, targets, markov_blanket, labels,
                           qy_logits, py_logits, z, zm, zv, zm_prior,
                           zv_prior)

  def get_multi_task_loss(self, task_batches):
    losses = []
    for task_name, batch in task_batches.items():
      inputs = batch[self.hp.inputs_key]
      if self.hp.loss_type == 'd':
        targets = None
      else:
        targets = batch[self.hp.targets_key]
      labels = {k: None for k in task_batches.keys()}
      labels[task_name] = batch[self.hp.labels_key]
      if self.hp.loss_reduce == "even":
        losses += [self.get_loss(task_name, labels, inputs,
                                 targets=targets,
                                 loss_type=self.hp.loss_type)]
      else:
        raise ValueError("bad loss combination type: %s" %
                         (self.hp.loss_reduce))
    return tf.add_n(losses)

  def get_loss(self, task, labels, inputs, targets=None,
               loss_type='gd'):
    # Keep track of batch length and size.
    self._batch_size = tf.shape(inputs)[0]

    # Encode the inputs using a task-specific encoder
    features = self.encode(inputs, task)

    disc = 'd' in loss_type
    gen = 'g' in loss_type
    if gen and disc:
      g_loss = self.get_total_generative_loss(labels, features, targets)
      d_loss = self.get_total_discriminative_loss(labels, features)
      a = self.hp.alpha
      return tf.reduce_mean((1. - a) * g_loss + (a * d_loss))
    elif disc:
      d_loss = self.get_total_discriminative_loss(labels, features)
      return tf.reduce_mean(d_loss)
    else:
      raise ValueError("unrecognized loss type: %s" % (loss_type))

  def encode(self, inputs, name):
    return self._encoders[name](inputs)

  def decode(self, targets, context, name):
    return self._decoders[name](targets, context)

  @property
  def hp(self):
    return self._hp

  @property
  def tau(self):
    return self._tau

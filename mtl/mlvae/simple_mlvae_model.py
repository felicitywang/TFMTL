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

from collections import OrderedDict

import tensorflow as tf

from tensorflow.contrib.training import HParams

from mtl.util.common import MLP_gaussian_posterior
from mtl.util.common import MLP_unnormalized_log_categorical

from mtl.vae.common import log_normal
from mtl.vae.common import gaussian_sample
from mtl.vae.common import get_tau

logging = tf.logging
tfd = tf.contrib.distributions


def default_hparams():
  return HParams(mlp_hidden_dim=512,
                 latent_dim=256,
                 tau0=0.5,
                 decay_tau=False,
                 alpha=0.5,
                 label_prior_type="uniform",
                 expectation="exact",
                 labels_key="label",
                 inputs_key="inputs",
                 targets_key="targets",
                 loss_reduce="even",
                 dtype='float32')


def generative_loss(nll_x, labels, qy_logits, py_logits, z, zm, zv,
                    zm_prior, zv_prior):
  nll_ys = []
  kl_ys = []
  for label_key, label_val in labels.items():
    qy_logit = qy_logits[label_key]
    py_logit = py_logits[label_key]
    if label_val is None:
      qcat = tfd.Categorical(logits=qy_logit,
                             name='qy_cat_{}'.format(label_key))
      pcat = tfd.Categorical(logits=py_logit,
                             name='py_cat_{}'.format(label_key))
      kl_ys += [tfd.kl_divergence(qcat, pcat)]
    else:
      nll_ys += [tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=py_logit,
        labels=label_val)]
  assert len(nll_ys) > 0
  assert len(kl_ys) > 0
  nll_y = tf.add_n(nll_ys)
  kl_y = tf.add_n(kl_ys)
  kl_z = log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
  return nll_x + nll_y + kl_y + kl_z


def tile_over_batch_dim(logits, batch_size):
  assert len(logits.get_shape().as_list()) == 1
  logits = tf.expand_dims(logits, 0)  # add batch dim
  return tf.tile(logits, [batch_size, 1])


class SimpleMultiLabelVAE(object):
  def __init__(self,
               class_sizes=None,
               encoders=None,
               decoders=None,
               hp=None):
    assert class_sizes is not None
    assert encoders.keys() == decoders.keys()
    assert class_sizes.keys() == encoders.keys()

    self._encoders = encoders
    self._decoders = decoders

    if hp is None:
      tf.logging.info("Using default hyper-parameters; none provided.")
      hp = default_hparams()
    self._hp = hp

    self._class_sizes = class_sizes

    # Intermediate loss values
    self._task_nll_x = dict()
    self._task_loss = dict()

    # p(y_1), ..., p(y_K)
    self._py_templates = dict()

    def uniform_prior(output_size):
      return tf.zeros([output_size])

    def learned_prior(output_size):
      return tf.get_variable('prior_weight', shape=[output_size],
                             trainable=True)

    if hp.label_prior_type == "uniform":
      prior_fn = uniform_prior
    elif hp.label_prior_type == "learned":
      prior_fn = learned_prior
    else:
      raise ValueError("unrecognized exception")

    for label_key, label_size in class_sizes.items():
      self._py_templates[label_key] = tf.make_template(
        'py_{}'.format(label_key),
        prior_fn,
        output_size=label_size)

    # p(z | y_1, ..., y_K)
    self._pz_template = tf.make_template('pz', MLP_gaussian_posterior,
                                         hidden_dim=hp.mlp_hidden_dim,
                                         latent_dim=hp.latent_dim)

    # q(y_1 | x), ..., q(y_K | x)
    self._qy_templates = dict()
    for k, v in class_sizes.items():
      self._qy_templates[k] = tf.make_template(
        'qy_{}'.format(k),
        MLP_unnormalized_log_categorical,
        output_size=v,
        hidden_dim=hp.mlp_hidden_dim)

    # q(z | y_1, ..., y_K)
    self._qz_template = tf.make_template('qz', MLP_gaussian_posterior,
                                         hidden_dim=hp.mlp_hidden_dim,
                                         latent_dim=hp.latent_dim)

    self._tau = get_tau(hp, decay=hp.decay_tau)

  def py_logits(self, label_key):
    return self._py_templates[label_key]()

  def qy_logits(self, label_key, features):
    return self._qy_templates[label_key](features)

  def pz_mean_var(self, ys):
    return self._pz_template(ys)

  def qz_mean_var(self, features, ys):
    return self._qz_template([features] + ys)

  def sample_y(self, logits, name):
    log_qy = tfd.ExpRelaxedOneHotCategorical(self._tau,
                                             logits=logits,
                                             name='log_qy_{}'.format(name))
    y = tf.exp(log_qy.sample())
    return y

  def get_predictions(self, inputs, feature_name):
    features = self.encode(inputs, feature_name)
    logits = self.qy_logits(feature_name, features)
    probs = tf.nn.softmax(logits)
    return tf.argmax(probs, axis=1)

  def get_task_discriminative_loss(self, task, labels, features):
    nll_ys = []
    for label_key, label_val in labels.items():
      if label_val is not None:
        nll_ys += [tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=label_val,
          logits=self.qy_logits(label_key, features),
          name='d_loss_{}'.format(label_key))]
    assert len(nll_ys)
    return tf.add_n(nll_ys)

  def get_task_generative_loss(self, task, labels, features, batch):
    ys = {}
    qy_logits = {}
    py_logits = {}
    batch_size = tf.shape(features)[0]
    self._latent_preds[task] = {}
    for label_key, label_val in labels.items():
      py_logits[label_key] = tile_over_batch_dim(self.py_logits(label_key),
                                                 batch_size)
      if label_val is None:
        qy_logits[label_key] = self.qy_logits(label_key, features)
        preds = tf.argmax(tf.nn.softmax(qy_logits[label_key]), axis=1)
        self._latent_preds[task][label_key] = preds
        ys[label_key] = self.sample_y(qy_logits[label_key], label_key)
      else:
        assert task not in self._obs_label
        self._obs_label[task] = label_val
        qy_logits[label_key] = None
        ys[label_key] = tf.one_hot(label_val, self._class_sizes[label_key])
    ys_list = ys.values()
    zm, zv = self.qz_mean_var(features, ys_list)
    z = gaussian_sample(zm, zv)
    zm_prior, zv_prior = self.pz_mean_var(ys_list)
    markov_blanket = tf.concat([z], axis=1)
    self._task_nll_x[task] = nll_x = self.decode(batch, markov_blanket, task)
    return generative_loss(nll_x, labels, qy_logits, py_logits, z, zm, zv,
                           zm_prior, zv_prior)

  def get_multi_task_loss(self, task_batches):
    losses = []
    self._g_loss = {}
    self._d_loss = {}
    self._latent_preds = {}
    self._obs_label = {}
    sorted_keys = sorted(task_batches.keys())  # order matters
    for task_name, batch in task_batches.items():
      labels = OrderedDict([(k, None) for k in sorted_keys])
      labels[task_name] = batch[self.hp.labels_key]
      if self.hp.loss_reduce == "even":
        with tf.name_scope(task_name):
          loss = self.get_loss(task_name, labels, batch)
          losses.append(loss)
          self._task_loss[task_name] = loss
      else:
        raise ValueError("bad loss combination type: %s" %
                         (self.hp.loss_reduce))
    return tf.add_n(losses, name='combined_mt_loss')

  def get_loss(self, task_name, labels, batch):
    # Encode the inputs using a task-specific encoder
    features = self.encode(batch, task_name)
    g_loss = self.get_task_generative_loss(task_name, labels,
                                           features, batch)
    assert len(g_loss.get_shape().as_list()) == 1
    self._g_loss[task_name] = g_loss = tf.reduce_mean(g_loss)
    d_loss = self.get_task_discriminative_loss(task_name, labels, features)
    assert len(d_loss.get_shape().as_list()) == 1
    self._d_loss[task_name] = d_loss = tf.reduce_mean(d_loss)
    a = self.hp.alpha
    assert a >= 0.0 and a <= 1.0, a
    return ((1. - a) * g_loss) + (a * d_loss)

  def get_generative_loss(self, task_name):
    return self._g_loss[task_name]

  def get_discriminative_loss(self, task_name):
    return self._d_loss[task_name]

  def obs_label(self, task_name):
    return self._obs_label[task_name]

  def latent_preds(self, task_name, label_name):
    return self._latent_preds[task_name][label_name]

  def encode(self, inputs, task_name):
    return self._encoders[task_name](inputs)

  def decode(self, targets, context, task_name):
    return self._decoders[task_name](targets, context)

  def get_task_loss(self, task_name):
    return self._task_loss[task_name]

  def get_task_nll_x(self, task_name):
    return self._task_nll_x[task_name]

  @property
  def hp(self):
    return self._hp

  @property
  def tau(self):
    return self._tau

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

import tensorflow as tf
from mtl.mlvae.vae import gaussian_sample
from mtl.mlvae.vae import get_tau
from mtl.mlvae.vae import log_normal
from mtl.util.common import MLP_gaussian_posterior
from mtl.util.common import MLP_unnormalized_log_categorical
from tensorflow.contrib.training import HParams

logging = tf.logging

Categorical = tf.contrib.distributions.Categorical
ExpRelaxedOneHotCategorical = tf.contrib.distributions.ExpRelaxedOneHotCategorical
kl_divergence = tf.contrib.distributions.kl_divergence


def default_hparams():
  return HParams(mlp_hidden_dim=512,
                 mlp_num_layers=2,
                 latent_dim=256,
                 tau0=0.5,
                 decay_tau=False,
                 alpha=0.5,
                 labels_key="label",
                 inputs_key="inputs",
                 targets_key="targets",
                 loss_reduce="even",
                 dtype='float32')


def tile_over_batch_dim(z, batch_size):
  assert len(z.get_shape().as_list()) == 1
  logits = tf.expand_dims(z, 0)  # add batch dim
  return tf.tile(logits, [batch_size, 1])


def generative_loss(nll_x, labels, qy_logits, py_logits, z, zm, zv,
                    zm_prior, zv_prior):
  nll_ys = []
  kl_ys = []
  for label_key, label_val in labels.items():
    qy_logit = qy_logits[label_key]
    py_logit = py_logits[label_key]
    if label_val is None:
      qcat = Categorical(logits=qy_logit, name='qy_cat_{}'.format(label_key))
      pcat = Categorical(logits=py_logit, name='py_cat_{}'.format(label_key))
      kl_ys += [kl_divergence(qcat, pcat)]
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


class MultiLabelVAE(object):
  def __init__(self,
               class_sizes=None,
               encoders=None,
               decoders=None,
               hp=None):
    assert class_sizes is not None
    assert encoders.keys() == decoders.keys()

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

    ########################### Generative Networks ###########################

    # p(z | task)
    def gaussian_prior():
      zm = tf.get_variable("mean", shape=[hp.latent_dim], trainable=True,
                           initializer=tf.zeros_initializer())
      zv_prm = tf.get_variable("var", shape=[hp.latent_dim], trainable=True,
                               initializer=tf.ones_initializer())
      zv = tf.nn.softplus(zv_prm)
      return (zm, zv)

    self._pz_template = dict()
    self._task_index = dict()
    i = 0
    for label_key in class_sizes:
      self._pz_template[label_key] = tf.make_template(
        'pz_{}'.format(label_key),
        gaussian_prior,
      )
      self._task_index[label_key] = i
      i += 1

    # p(y_1 | z), ..., p(y_K | z)
    self._py_templates = dict()
    for label_key, label_size in class_sizes.items():
      self._py_templates[label_key] = tf.make_template(
        'py_{}'.format(label_key),
        MLP_unnormalized_log_categorical,
        output_size=label_size,
        hidden_dim=hp.mlp_hidden_dim,
        num_layers=hp.mlp_num_layers)

    ########################### Inference Networks ############################

    # q(y_1 | x, z), ..., q(y_K | x, z)
    self._qy_templates = dict()
    for k, v in class_sizes.items():
      self._qy_templates[k] = tf.make_template(
        'qy_{}'.format(k),
        MLP_unnormalized_log_categorical,
        output_size=v,
        hidden_dim=hp.mlp_hidden_dim,
        num_layers=hp.mlp_num_layers)

    # q(z | x, task)
    self._qz_template = tf.make_template('qz',
                                         MLP_gaussian_posterior,
                                         latent_dim=hp.latent_dim,
                                         hidden_dim=hp.mlp_hidden_dim,
                                         num_layers=hp.mlp_num_layers)

    self._tau = get_tau(hp, decay=hp.decay_tau)

  def py_logits(self, label_key, z):
    return self._py_templates[label_key](z)

  def qy_logits(self, label_key, features, z):
    return self._qy_templates[label_key]([features, z])

  def pz_mean_var(self, task):
    return self._pz_template[task]()

  def task_vec(self, task):
    num_tasks = len(self._task_index)
    task_idx = self._task_index[task]
    return tf.one_hot(task_idx, num_tasks)

  def qz_mean_var(self, task, features):
    batch_size = tf.shape(features)[0]
    task_vec = self.task_vec(task)
    tiled_task_vec = tile_over_batch_dim(task_vec, batch_size)
    return self._qz_template([features, tiled_task_vec])

  def sample_y(self, logits, name):
    log_qy = ExpRelaxedOneHotCategorical(self._tau,
                                         logits=logits,
                                         name='log_qy_{}'.format(name))
    y = tf.exp(log_qy.sample())
    return y

  def get_predictions(self, inputs, task):
    features = self.encode(inputs, task)
    zm, _ = self.qz_mean_var(task, features)
    logits = self.qy_logits(task, features, zm)
    probs = tf.nn.softmax(logits)
    return tf.argmax(probs, axis=1)

  def get_multi_task_loss(self, task_batches):
    losses = []
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
    features = self.encode(batch, task_name)
    self._latent_preds[task_name] = {}
    ys = {}
    qy_logits = {}
    py_logits = {}
    batch_size = tf.shape(features)[0]
    zm, zv = self.qz_mean_var(task_name, features)
    z = gaussian_sample(zm, zv)
    zm_prior, zv_prior = self.pz_mean_var(task_name)
    zm_prior = tile_over_batch_dim(zm_prior, batch_size)
    zv_prior = tile_over_batch_dim(zv_prior, batch_size)

    for label_key, label_val in labels.items():
      py_logits[label_key] = self.py_logits(label_key, z)
      if label_val is None:
        qy_logits[label_key] = self.qy_logits(label_key, features, z)
        preds = tf.argmax(tf.nn.softmax(qy_logits[label_key]), axis=1)
        self._latent_preds[task_name][label_key] = preds
        ys[label_key] = self.sample_y(qy_logits[label_key], label_key)
      else:
        assert task_name not in self._obs_label
        self._obs_label[task_name] = label_val
        qy_logits[label_key] = None
        ys[label_key] = tf.one_hot(label_val, self._class_sizes[label_key])

    ys_list = ys.values()
    markov_blanket = tf.concat(ys_list, axis=1)
    self._task_nll_x[task_name] = nll_x = self.decode(batch, markov_blanket,
                                                      task_name)
    g_loss = generative_loss(nll_x, labels, qy_logits, py_logits, z,
                             zm, zv, zm_prior, zv_prior)
    assert len(g_loss.get_shape().as_list()) == 1
    self._g_loss = g_loss = tf.reduce_mean(g_loss)
    nll_ys = []
    for label_key, label_val in labels.items():
      if label_val is not None:
        nll_ys += [tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=label_val,
          logits=self.qy_logits(label_key, features, z),
          name='d_loss_{}'.format(label_key))]
    assert len(nll_ys)
    d_loss = tf.add_n(nll_ys, name='sum_d_loss_{}'.format(task_name))
    assert len(d_loss.get_shape().as_list()) == 1
    self._d_loss = d_loss = tf.reduce_mean(d_loss)
    a = self.hp.alpha
    assert a >= 0.0 and a <= 1.0, a
    return (1. - a) * g_loss + (a * d_loss)

  def encode(self, inputs, task_name):
    return self._encoders[task_name](inputs)

  def decode(self, targets, context, task_name):
    return self._decoders[task_name](targets, context)

  def get_task_loss(self, task_name):
    return self._task_loss[task_name]

  def get_task_nll_x(self, task_name):
    return self._task_nll_x[task_name]

  def get_generative_loss(self, task_name):
    return self._g_loss[task_name]

  def get_discriminative_loss(self, task_name):
    return self._d_loss[task_name]

  def obs_label(self, task_name):
    return self._obs_label[task_name]

  def latent_preds(self, task_name, label_name):
    return self._latent_preds[task_name][label_name]

  @property
  def hp(self):
    return self._hp

  @property
  def tau(self):
    return self._tau

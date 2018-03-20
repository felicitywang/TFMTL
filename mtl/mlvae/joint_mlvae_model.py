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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from operator import mul
from operator import itemgetter
from itertools import product
from collections import OrderedDict

import tensorflow as tf

from tensorflow.contrib.training import HParams

from mtl.layers import dense_layer
from mtl.layers import mlp

from mtl.mlvae.prob import enum_events
from mtl.mlvae.prob import entropy
from mtl.mlvae.prob import normalize_logits
from mtl.mlvae.prob import marginal_log_prob
from mtl.mlvae.prob import conditional_log_prob

from mtl.mlvae.vae import log_normal
from mtl.mlvae.vae import gaussian_sample
from mtl.mlvae.vae import get_tau

logging = tf.logging
tfd = tf.contrib.distributions

def default_hparams():
  return HParams(qy_mlp_hidden_dim=512,
                 qy_mlp_num_layer=2,
                 qz_mlp_hidden_dim=512,
                 qz_mlp_num_layer=2,
                 pz_mlp_hidden_dim=512,
                 pz_mlp_num_layer=0,
                 latent_dim=256,
                 tau0=0.5,
                 layer_norm=True,
                 decay_tau=False,
                 alpha=0.5,
                 y_prediction="deterministic",
                 label_prior_type="uniform",
                 y_inference="exact",
                 labels_key="label",
                 inputs_key="inputs",
                 targets_key="targets",
                 loss_reduce="even",
                 dtype='float32')


def sampled_generative_loss(nll_x, labels, qy_logits, py_logits, z, zm, zv,
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


def joint_posterior_logits(x, output_size, **kwargs):
  x = mlp(x, **kwargs)
  return dense_layer(x, output_size, 'logits', activation=None)


def gaussian_posterior(x, latent_dim, **kwargs):
  x = mlp(x, **kwargs)
  zm = dense_layer(x, latent_dim, 'zm', activation=None)
  zv = dense_layer(x, latent_dim, 'zv', activation=tf.nn.softplus)
  return zm, zv


class JointMultiLabelVAE(object):
  def __init__(self,
               is_training=True,
               prior=None,
               class_sizes=None,
               encoders=None,
               decoders=None,
               hp=None):

    assert class_sizes is not None
    assert encoders.keys() == decoders.keys()
    assert class_sizes.keys() == encoders.keys()

    if type(class_sizes) is dict:
      self._class_sizes = OrderedDict(sorted(class_sizes.items(),
                                             key=itemgetter(1)))
    elif type(class_sizes) is OrderedDict:
      self._class_sizes = class_sizes
    else:
      raise ValueError("class_sizes must be of type dict")

    tf.logging.info("Class sizes:")
    for k, v in self._class_sizes.items():
      tf.logging.info("  %s: %d", k, v)

    self._encoders = encoders
    self._decoders = decoders

    if hp is None:
      tf.logging.info("Using default hyper-parameters; none provided.")
      hp = default_hparams()
    self._hp = hp

    # Intermediate loss values
    self._task_nll_x = dict()
    self._task_loss = dict()

    ########################### Generative Networks ###########################

    output_size = reduce(mul, class_sizes.values())
    tf.logging.info("Full size of event space of joint distribution: %d",
                    output_size)

    # ln p(y_1, ..., y_K)

    def fixed_prior():
      if len(prior.shape) == 1:
        # Joint prior ln p(y_1, ..., y_K) as one vector where, for
        # two labels Y1 and Y2 with values {a, b} and {1, 2}:
        #
        #  prior[0] = ln p(Y1 = a, Y2 = 1)
        #  prior[1] = ln p(Y1 = a, Y2 = 2)
        #  prior[2] = ln p(Y1 = b, Y2 = 1)
        #  prior[4] = ln p(Y1 = b, Y2 = 2)
        #
        assert prior.shape[0] == output_size
        return tf.constant(prior)
      elif len(prior.shape) == 2:
        # Independent priors: ln p(y_1), ..., ln p(y_K)
        raise ValueError("unimplemented")
        # flat_prior = np.zeros([output_size])
        # sizes = class_sizes.values()
        # iters = [xrange(R) for R in sizes]
        # i = 0
        # for idx in product(iters):
        #   flat_prior[i] = reduce(sum,
        #                          [np.log(x) for x in prior[idx]])
        #   i += 1
      else:
        raise ValueError("bad prior")

    def uniform_prior():
      return tf.ones([output_size])

    def learned_prior():
      return tf.get_variable('prior_weight', shape=[output_size],
                             trainable=True)

    if hp.label_prior_type == "uniform":
      prior_fn = uniform_prior
    elif hp.label_prior_type == "learned":
      prior_fn = learned_prior
    elif hp.label_prior_type == "fixed":
      if prior is None:
        raise ValueError("specified prior type but no prior given")
      prior_fn = fixed_prior
    else:
      raise ValueError("unrecognized exception")

    self._ln_py_template = tf.make_template('py', prior_fn,
                                            output_size=output_size)

    # ln p(z | y_1, ..., y_K)
    self._ln_pz_template = tf.make_template('pz',
                                            gaussian_posterior,
                                            latent_dim=hp.latent_dim,
                                            hidden_dim=hp.pz_mlp_hidden_dim,
                                            num_layer=hp.pz_mlp_num_layer)

    ########################### Inference Networks ############################

    # ln q(y_1, ..., y_K | x)
    self._ln_qy_template = tf.make_template('qy',
                                            joint_posterior_logits,
                                            output_size=output_size,
                                            hidden_dim=hp.qy_mlp_hidden_dim,
                                            num_layer=hp.qy_mlp_num_layer)

    # ln q(z | x, y_1, ..., y_K)
    self._ln_qz_template = tf.make_template('qz',
                                            gaussian_posterior,
                                            latent_dim=hp.latent_dim,
                                            hidden_dim=hp.qz_mlp_hidden_dim,
                                            num_layer=hp.qz_mlp_num_layer)

    # NOTE: In general we will probably use a constant value for tau.
    self._tau = get_tau(hp, decay=hp.decay_tau)

  def label_index(self, label):
    index = 0
    for k in self.class_sizes.keys():
      if k == label:
        return index
      index += 1
    raise ValueError("unknown label key: %s" % label)

  def joint_logits(self, x):
    return self._ln_qy_template(x)

  def log_joint_prob(self, x):
    logits = self.joint_logits(x)
    class_dims = self.class_sizes.values()
    return normalize_logits(logits, dims=class_dims)

  def py_logits(self, label_key):
    if self.hp.label_prior_type == "uniform":
      label_dim = self.class_sizes[label_key]
      return tf.log(tf.constant([1./label_dim] * label_dim))
    else:
      raise ValueError("unimplemented")

  def qy_given_x_logits(self, log_joint_normalized, target_index, cond_index,
                        cond_val=None):
    batch_size = tf.shape(log_joint_normalized)[0]
    #tf.logging.info('log joint normalized: %s', log_joint_normalized)
    logits = conditional_log_prob(log_joint_normalized, target_index,
                                  cond_index)
    #tf.logging.info('target index: %d', target_index)
    #tf.logging.info('cond index: %s', cond_index)
    if cond_val is not None:
      with tf.name_scope('conditioning'):
        # TODO(noa): test case this
        final_dim = logits.get_shape()[-1]
        indices = tf.stack([tf.range(batch_size),
                            tf.to_int32(cond_val)], axis=1)
        logits = tf.gather_nd(logits, indices)
        assert len(logits.get_shape()) == 2
    else:
      assert len(logits.get_shape()) == 3

    return logits

  def qy_logits(self, log_joint_normalized, target_index):
    return marginal_log_prob(log_joint_normalized, target_index)

  def pz_mean_var(self, ys):
    assert type(ys) is list
    ys_concat = tf.concat(ys, axis=1)
    return self._ln_pz_template(ys_concat)

  def qz_mean_var(self, features, ys):
    assert type(ys) is list
    xy = tf.concat([features] + ys, axis=1)
    return self._ln_qz_template(xy)

  def sample_y(self, logits, name):
    with tf.name_scope('sample'):
      log_qy = tfd.ExpRelaxedOneHotCategorical(self._tau,
                                               logits=logits,
                                               name='log_qy_{}'.format(name))
      y = tf.exp(log_qy.sample())
      return y

  def get_predictions(self, inputs, feature_name):
    features = self.encode(inputs, feature_name)
    log_joint = self.log_joint_prob(features)
    target_axis = self.label_index(feature_name)
    logits = self.qy_logits(log_joint, target_axis)
    probs = tf.nn.softmax(logits)
    return tf.argmax(probs, axis=1)

  def get_task_discriminative_loss(self, task, labels, log_joint):
    nll_ys = []
    for label_key, label_val in labels.items():
      label_axis = self.label_index(label_key)
      if label_val is not None:
        nll_ys += [tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=label_val,
          logits=self.qy_logits(log_joint, label_axis),
          name='d_loss_{}'.format(label_key))]
    assert len(nll_ys)
    return tf.add_n(nll_ys)

  def get_exact_generative_loss(self, task_name, labels, features,
                                log_joint, batch):
    assert type(labels) is OrderedDict
    self._latent_preds[task_name] = {}
    ys = {}
    qy_logits = {}
    py_logits = {}
    batch_size = tf.shape(features)[0]

    # Dimension for the observed label
    observed_axis = None
    observed_label = None
    for k, v in labels.items():
      if v is None:
        pass
      else:
        assert observed_axis is None, "assume one observed feature"
        observed_axis = self.label_index(k)
        observed_label = v

    # Set up observed / latent variables
    obs_label = None
    for label_key, label_val in labels.items():
      py_logits[label_key] = tile_over_batch_dim(self.py_logits(label_key),
                                                 batch_size)
      if label_val is None:
        target_axis = self.label_index(label_key)
        qy_logits[label_key] = self.qy_given_x_logits(log_joint,
                                                      target_axis,
                                                      observed_axis,
                                                      cond_val=observed_label)
        preds = tf.argmax(tf.nn.softmax(qy_logits[label_key]), axis=1)
        self._latent_preds[task_name][label_key] = preds
      else:
        assert label_key not in self._obs_label
        assert label_key
        self._obs_label[task_name] = obs_label = label_val

    # Loop over all possible events conditioning on observed ones
    assert obs_label is not None
    assert type(self.class_sizes) is OrderedDict, "need ordered dict"
    events = enum_events(self.class_sizes, cond_vals={task_name: obs_label})

    def labeled_loss(e):
      assert type(e) is list
      ys = OrderedDict(zip(labels.keys(), e))
      onehot_ys = []
      for k, v in ys.items():
        size = self.class_sizes[k]
        onehot_ys += [tf.one_hot(v, size)]
      assert len(ys) == len(py_logits)
      assert len(ys) == len(qy_logits) + 1
      nll_ys = []
      for k, label in ys.items():
        assert type(label) is tf.Tensor
        nll_ys += [tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=py_logits[k],
          labels=label
        )]
      assert len(nll_ys) > 0
      zm, zv = self.qz_mean_var(features, onehot_ys)
      z = gaussian_sample(zm, zv)
      zm_prior, zv_prior = self.pz_mean_var(onehot_ys)
      nll_y = tf.add_n(nll_ys)
      kl_z = log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
      nll_x = self.decode(batch, z, task_name)
      return nll_x + nll_y + kl_z

    # Compute the losses over all label combinations
    batch_size = tf.shape(features)[0]
    losses = []
    for e in events:
      nll_qys = []
      i = 0
      for label_key, label_val in labels.items():
        if label_val is None:  # latent variable
          enum_val = e[i]
          nll_qys += [tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=qy_logits[label_key],
            labels=enum_val
          )]
        i += 1
      assert len(nll_qys) > 0
      nll_qy = tf.add_n(nll_qys)
      qy_prob = tf.exp(nll_qy)
      losses += [qy_prob * labeled_loss(e)]

    # Compute H(q(y))
    entropies = []
    for k, v in labels.items():
      if v is None:
        entropies.append(entropy(qy_logits[k]))
    assert len(entropies) > 0
    self._label_entropy = label_entropy = tf.add_n(entropies)
    g_loss = tf.add_n(losses + [-label_entropy])
    assert len(g_loss.get_shape().as_list()) == 1
    self._g_loss[task_name] = g_loss = tf.reduce_mean(g_loss)
    return g_loss

  def get_sample_generative_loss(self, task, labels, features, log_joint,
                                 batch):
    ys = {}
    qy_logits = {}
    py_logits = {}
    batch_size = tf.shape(features)[0]

    # Dimension for the observed label
    observed_axis = None
    observed_label = None
    for k, v in labels.items():
      if v is None:
        pass
      else:
        assert observed_axis is None, "assume one observed feature"
        observed_axis = self.label_index(k)
        observed_label = v

    # Accumulate predicted or observed labels
    self._latent_preds[task] = {}
    for label_key, label_val in labels.items():
      py_logits[label_key] = tile_over_batch_dim(self.py_logits(label_key),
                                                 batch_size)
      if label_val is None:
        target_axis = self.label_index(label_key)
        qy_logits[label_key] = self.qy_given_x_logits(log_joint, target_axis,
                                                      observed_axis,
                                                      cond_val=observed_label)
        preds = tf.argmax(tf.nn.softmax(qy_logits[label_key]), axis=1)
        self._latent_preds[task][label_key] = preds
        ys[label_key] = self.sample_y(qy_logits[label_key], label_key)
      else:
        assert label_key not in self._obs_label
        assert label_key == task
        self._obs_label[label_key] = label_val
        qy_logits[label_key] = None
        ys[label_key] = tf.one_hot(label_val, self.class_sizes[label_key])

    # p(z | ys) and q(z | x, ys)
    tf.logging.info('Labels:')
    for k, v in ys.items():
      tf.logging.info('  %s:  %s', k, v)
    ys_list = ys.values()
    zm, zv = self.qz_mean_var(features, ys_list)
    z = gaussian_sample(zm, zv)
    zm_prior, zv_prior = self.pz_mean_var(ys_list)

    self._task_nll_x[task] = nll_x = self.decode(batch, z, task)
    return sampled_generative_loss(nll_x, labels, qy_logits,
                                   py_logits, z, zm, zv, zm_prior,
                                   zv_prior)

  def get_multi_task_loss(self, task_batches, unlabeled_batches=None):
    losses = []
    self._obs_label = {}
    self._latent_preds = {}
    self._d_loss = {}
    self._g_loss = {}
    self._unlabeled_task_loss = {}
    sorted_keys = sorted(task_batches.keys())
    for task_name, batch in task_batches.items():
      labels = OrderedDict([(k, None) for k in sorted_keys])
      labels[task_name] = batch[self.hp.labels_key]
      if self.hp.loss_reduce == "even" or self.hp.loss_reduce == "scaled":
        with tf.name_scope(task_name):
          loss = self.get_loss(task_name, labels, batch)
          losses.append(loss)
          self._task_loss[task_name] = loss
          # if unlabeled_batches and task_name in unlabeled_batches:
          #   tf.logging.info("Unlabeled batch: %s", task_name)
          #   unlabeled_batch = unlabeled_batches[task_name]
          #   loss = self.get_unlabeled_loss(task_name, unlabeled_batch)
          #   self._unlabeled_task_loss[task_name] = loss
          #   losses.append(loss)
      else:
        raise ValueError("bad loss combination type: %s" %
                         (self.hp.loss_reduce))

    total_loss = tf.add_n(losses, name='combined_multi_task_loss')

    if self.hp.loss_reduce == "even":
      return total_loss / float(len(losses))
    else:
      raise ValueError('unimplemented')

  def get_unlabeled_loss(self, task_name, batch):
    # Encode the inputs using a task-specific encoder
    features = self.encode(batch, task_name)

    # Get normalized log q(y_1, ..., y_K | x)
    logits = self.joint_logits(features)
    class_dims = self.class_sizes.values()
    log_joint = normalize_logits(logits, dims=class_dims)

    raise ValueError("unimplemented")

    if self.hp.y_inference == "exact":
      raise ValueError("unimplemented")
    else:
      g_loss = self.get_sample_generative_loss(task_name, labels,
                                               features,
                                               log_joint, batch)

    assert len(g_loss.get_shape().as_list()) == 1
    self._u_loss[task_name] = g_loss = tf.reduce_mean(g_loss)

  def get_loss(self, task_name, labels, batch):
    # Encode the inputs using a task-specific encoder
    features = self.encode(batch, task_name)

    # Get normalized log q(y_1, ..., y_K | x)
    logits = self.joint_logits(features)
    class_dims = self.class_sizes.values()
    log_joint = normalize_logits(logits, dims=class_dims)

    if self.hp.y_inference == "exact":
      g_loss = self.get_exact_generative_loss(task_name, labels,
                                              features, log_joint, batch)
    elif self.hp.y_inference == "sample":
      g_loss = self.get_sample_generative_loss(task_name, labels,
                                               features,
                                               log_joint, batch)
    else:
      raise ValueError("unrecognized inference mode: %s" % self.hp.y_inference)

    d_loss = self.get_task_discriminative_loss(task_name, labels,
                                               log_joint)
    assert len(d_loss.get_shape().as_list()) == 1
    self._d_loss[task_name] = d_loss = tf.reduce_mean(d_loss)
    a = self.hp.alpha
    assert a >= 0.0 and a <= 1.0, a
    return (1. - a) * g_loss + (a * d_loss)

  def encode(self, inputs, task_name):
    return self._encoders[task_name](inputs)

  def decode(self, targets, context, task_name):
    return self._decoders[task_name](targets, context)

  def get_task_loss(self, task_name):
    return self._task_loss[task_name]

  def get_generative_loss(self, task_name):
    return self._g_loss[task_name]

  def get_discriminative_loss(self, task_name):
    return self._d_loss[task_name]

  def get_task_nll_x(self, task_name):
    return self._task_nll_x[task_name]

  def obs_label(self, task_name):
    return self._obs_label[task_name]

  def latent_preds(self, task_name, label_name):
    return self._latent_preds[task_name][label_name]

  @property
  def class_sizes(self):
    return self._class_sizes

  @property
  def hp(self):
    return self._hp

  @property
  def tau(self):
    return self._tau

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
from operator import itemgetter

import tensorflow as tf

from mtl.layers import dense_layer
from mtl.layers import mlp

from mtl.vae.prob import enum_events
from mtl.vae.prob import entropy

from mtl.vae.common import gaussian_sample
from mtl.vae.common import get_tau
from mtl.vae.common import log_normal

from tensorflow.contrib.training import HParams

logging = tf.logging
tfd = tf.contrib.distributions


def default_hparams():
    return HParams(py_mlp_hidden_dim=512,
                   py_mlp_num_layer=0,
                   qy_mlp_hidden_dim=512,
                   qy_mlp_num_layer=2,
                   qz_mlp_hidden_dim=512,
                   qz_mlp_num_layer=2,
                   latent_dim=256,
                   tau0=0.5,
                   decay_tau=False,
                   alpha=0.5,
                   y_inference="exact",
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


def prior_logits(x, output_size, **kwargs):
    x = mlp(x, **kwargs)
    return dense_layer(x, output_size, 'logits', activation=None)


def posterior_logits(x, output_size, **kwargs):
    x = mlp(x, **kwargs)
    return dense_layer(x, output_size, 'logits', activation=None)


def posterior_gaussian(x, latent_dim, **kwargs):
    x = mlp(x, **kwargs)
    zm = dense_layer(x, latent_dim, 'zm', activation=None)
    zv = dense_layer(x, latent_dim, 'zv', activation=tf.nn.softplus)
    return zm, zv


class MultiLabelVAE(object):
    def __init__(self,
                 is_training=True,
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

        # Intermediate loss values
        self._task_nll_x = dict()
        self._task_loss = dict()

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
                prior_logits,
                is_training=is_training,
                output_size=label_size,
                hidden_dims=hp.py_mlp_hidden_dim,
                num_layers=hp.py_mlp_num_layer)

        # q(y_1 | z), ..., q(y_K | z)
        self._qy_templates = dict()
        for k, v in class_sizes.items():
            self._qy_templates[k] = tf.make_template(
                'qy_{}'.format(k),
                posterior_logits,
                is_training=is_training,
                output_size=v,
                hidden_dims=hp.qy_mlp_hidden_dim,
                num_layers=hp.qy_mlp_num_layer)

        # q(z | x, task)
        self._qz_template = tf.make_template('qz',
                                             posterior_gaussian,
                                             is_training=is_training,
                                             latent_dim=hp.latent_dim,
                                             hidden_dims=hp.qz_mlp_hidden_dim,
                                             num_layers=hp.qz_mlp_num_layer)

        self._tau = get_tau(hp, decay=hp.decay_tau)

    def py_logits(self, label_key, z):
        return self._py_templates[label_key](z)

    def qy_logits(self, label_key, features, z):
        cond = tf.concat([features, z], axis=1)
        return self._qy_templates[label_key](cond)

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
        cond = tf.concat([features, tiled_task_vec], axis=1)
        return self._qz_template(cond)

    def sample_y(self, logits, name):
        with tf.name_scope('sample'):
            log_qy = tfd.ExpRelaxedOneHotCategorical(self._tau,
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

    def get_multi_task_loss(self, task_batches, unlabeled_batches=None):
        losses = []
        self._latent_preds = {}
        self._obs_label = {}
        self._g_loss = {}
        self._d_loss = {}
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

        total_loss = tf.add_n(losses, name='combined_mt_loss')

        if self.hp.loss_reduce == "even":
            return total_loss / float(len(losses))
        else:
            raise ValueError('unimplemented')

    def get_loss(self, task_name, labels, batch):
        features = self.encode(batch, task_name)

        # Generative loss
        if self.hp.y_inference == "exact":
            g_loss = self.get_exact_generative_loss(task_name, labels, batch,
                                                    features)
        elif self.hp.y_inference == "sample":
            g_loss = self.get_sampled_generative_loss(task_name, labels, batch,
                                                      features)
        else:
            raise ValueError("unrecognized inference mode: %s" % self.hp.y_inference)

        # Discriminative loss
        nll_ys = []
        zm, zv = self.qz_mean_var(task_name, features)
        z = gaussian_sample(zm, zv)
        for label_key, label_val in labels.items():
            if label_val is not None:
                nll_ys += [tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=label_val,
                    logits=self.qy_logits(label_key, features, z),
                    name='d_loss_{}'.format(label_key))]
        assert len(nll_ys)
        d_loss = tf.add_n(nll_ys, name='sum_d_loss_{}'.format(task_name))
        assert len(d_loss.get_shape().as_list()) == 1
        self._d_loss[task_name] = d_loss = tf.reduce_mean(d_loss)
        a = self.hp.alpha
        assert a >= 0.0 and a <= 1.0, a
        return (1. - a) * g_loss + (a * d_loss)

    def get_exact_generative_loss(self, task_name, labels, batch, features):
        assert type(labels) is OrderedDict
        self._latent_preds[task_name] = {}

        qy_logits = {}
        py_logits = {}
        batch_size = tf.shape(features)[0]
        zm, zv = self.qz_mean_var(task_name, features)
        z = gaussian_sample(zm, zv)
        zm_prior, zv_prior = self.pz_mean_var(task_name)
        zm_prior = tile_over_batch_dim(zm_prior, batch_size)
        zv_prior = tile_over_batch_dim(zv_prior, batch_size)

        # Set up observed / latent variables
        obs_label = None
        for label_key, label_val in labels.items():
            py_logits[label_key] = self.py_logits(label_key, z)
            if label_val is None:
                qy_logits[label_key] = self.qy_logits(label_key, features, z)
                preds = tf.argmax(tf.nn.softmax(qy_logits[label_key]), axis=1)
                self._latent_preds[task_name][label_key] = preds
            else:
                assert label_key not in self._obs_label
                assert label_key
                self._obs_label[task_name] = obs_label = label_val

        # Loop over all possible events conditioning on observed ones
        assert obs_label is not None
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
            nll_y = tf.add_n(nll_ys)
            kl_z = log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
            markov_blanket = tf.concat(onehot_ys, axis=1)
            nll_x = self.decode(batch, markov_blanket, task_name)
            return nll_x + nll_y + kl_z

        losses = []
        batch_size = tf.shape(features)[0]
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
            nll_qy = tf.add_n(nll_qys)  # qy logits are independent
            qy_prob = tf.exp(nll_qy)  # TODO(noa): negative sign here?
            losses += [qy_prob * labeled_loss(e)]
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

    def get_sampled_generative_loss(self, task_name, labels, batch, features):
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
        self._g_loss[task_name] = g_loss = tf.reduce_mean(g_loss)
        return g_loss

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
    def class_sizes(self):
        return self._class_sizes

    @property
    def hp(self):
        return self._hp

    @property
    def tau(self):
        return self._tau

    @property
    def label_entropy(self):
        return self._label_entropy

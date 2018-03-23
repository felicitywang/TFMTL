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

from mtl.encoders.encoder_factory import build_encoders
from mtl.layers.mlp import dense_layer, mlp

logging = tf.logging


def validate_labels(feature_dict, class_sizes):
    for k in feature_dict:
        with tf.control_dependencies([
            tf.assert_greater(tf.cast(class_sizes[k], dtype=tf.int64),
                              tf.cast(tf.reduce_max(feature_dict[k]),
                                      dtype=tf.int64))]):
            pass


class Mult(object):
    def __init__(self,
                 class_sizes=None,
                 dataset_order=None,
                 vocab_size=None,
                 encoder_hps=None,
                 # TODO keep one of hps and args only
                 hps=None,
                 args=None):

        # class_sizes: map from feature names to cardinality of label sets
        # dataset_order: list of features in some fixed order
        #   (concatenated order matters for decoding)
        # encoders: one per dataset

        assert class_sizes is not None
        assert dataset_order is not None
        assert vocab_size is not None
        assert hps is not None
        assert args is not None

        self._class_sizes = class_sizes
        self._dataset_order = dataset_order
        self._encoder_hps = encoder_hps
        self._hps = hps
        # self._args = args

        # all feature names are present and consistent across data structures
        assert set(class_sizes.keys()) == set(dataset_order)

        self._encoders = build_encoders(vocab_size, args, encoder_hps)

        self._mlps_shared = build_mlps(hps, is_shared=True)
        self._mlps_private = build_mlps(hps, is_shared=False)

        self._logits = build_logits(class_sizes)

    # Encoding (feature extraction)
    def encode(self, inputs, dataset_name, lengths=None):
        # if self._encoders[dataset_name] == 'no_op':
        #     return inputs
        return self._encoders[dataset_name](inputs, lengths)

    def get_predictions(self, batch, dataset_name):
        # Returns most likely label given conditioning variables (only
        # run this on eval data)
        x = batch[self._hps.input_key]
        input_lengths = batch[self._hps.token_lengths_key]

        x = self.encode(x, dataset_name, lengths=input_lengths)

        x = self._mlps_shared[dataset_name](x, is_training=False)
        x = self._mlps_private[dataset_name](x, is_training=False)

        x = self._logits[dataset_name](x)

        res = tf.argmax(x, axis=1)
        # res = tf.expand_dims(res, axis=1)

        return res

    def get_loss(self, batch, dataset_name, features=None, is_training=True):
        # Returns most likely label given conditioning variables (only
        # run this on eval data)
        x = batch[self._hps.input_key]
        input_lengths = batch[self._hps.token_lengths_key]
        labels = batch[self._hps.label_key]

        # TODO remove this?
        if features is None:
            x = self.encode(x, dataset_name, lengths=input_lengths)
        else:
            x = features

        x = self._mlps_shared[dataset_name](x, is_training)
        x = self._mlps_private[dataset_name](x, is_training)

        x = self._logits[dataset_name](x)

        # loss
        ce = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=x, labels=tf.cast(labels, dtype=tf.int32)))

        # l2 regularization
        variables = tf.trainable_variables()
        l2_weight_penalty = tf.add_n([
            tf.nn.l2_loss(v) for v in variables if 'bias' not in v.name
        ]) * self._hps.l2_weight

        loss = tf.reduce_mean(ce) + l2_weight_penalty

        return loss

    def get_multi_task_loss(self, dataset_batches, is_training):
        # dataset_batches: map from dataset names to training batches
        # (one batch per dataset); we assume only one dataset's labels
        # are observed; the rest are unobserved

        losses = dict()
        total_loss = 0.0
        assert len(dataset_batches) == len(self._hps.alphas)
        assert sum(self._hps.alphas) == 1.0
        for dataset_batch, alpha in zip(dataset_batches.items(),
                                        self._hps.alphas):
            # We assume that the encoders and decoders always use the
            # same fields/features (given by the keys in the batch
            # accesses below)
            dataset_name, batch = dataset_batch
            loss = self.get_loss(batch=batch, dataset_name=dataset_name,
                                 features=None, is_training=is_training)
            total_loss += alpha * loss
            losses[dataset_name] = loss

        return total_loss

    @property
    def encoders(self):
        return self._encoders

    @property
    def hp(self):
        return self._hps


def build_mlps(hps, is_shared):
    mlps = dict()
    if is_shared:
        mlp_shared = tf.make_template('mlp_shared',
                                      mlp,
                                      hidden_dims=hps.shared_hidden_dims,
                                      num_layers=hps.shared_mlp_layers,
                                      # TODO from args
                                      activation=tf.nn.relu,
                                      input_keep_prob=hps.input_keep_prob,
                                      # TODO ?
                                      batch_normalization=False,
                                      # TODO ?
                                      layer_normalization=False,
                                      # TODO ?
                                      output_keep_prob=1,
                                      )
        for dataset in hps.datasets:
            mlps[dataset] = mlp_shared
        return mlps
    else:
        for dataset in hps.datasets:
            mlps[dataset] = tf.make_template(
                'mlp_{}'.format(dataset),
                mlp,
                hidden_dims=hps.private_hidden_dims,
                num_layers=hps.private_mlp_layers,
                # TODO from args
                activation=tf.nn.relu,
                # TODO args.dropout_rate to keep_prob
                input_keep_prob=1,
                # TODO ?
                batch_normalization=False,
                # TODO ?
                layer_normalization=False,
                output_keep_prob=hps.output_keep_prob,
            )

    return mlps


def build_logits(class_sizes):
    logits = dict()
    for k, v in class_sizes.items():
        logits[k] = tf.make_template('logit_{}'.format(k),
                                     dense_layer,
                                     name='logits',
                                     output_size=v,
                                     activation=None
                                     )
    return logits

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

logging = tf.logging


def validate_labels(feature_dict, class_sizes):
    for k in feature_dict:
        with tf.control_dependencies([
            tf.assert_greater(tf.cast(class_sizes[k], dtype=tf.int64),
                              tf.cast(tf.reduce_max(feature_dict[k]), dtype=tf.int64)
                              )]):
            pass


class Mult(object):
    def __init__(self,
                 class_sizes=None,
                 dataset_order=None,
                 encoders=None,
                 mlps=None,
                 hps=None):

        # class_sizes: map from feature names to cardinality of label sets
        # dataset_order: list of features in some fixed order
        #   (concatenated order matters for decoding)
        # encoders: one per dataset

        assert class_sizes is not None
        assert dataset_order is not None
        assert encoders is not None
        assert mlps is not None
        assert hps is not None

        self._hps = hps

        self._class_sizes = class_sizes

        self._dataset_order = dataset_order

        assert set(class_sizes.keys()) == set(
            dataset_order)  # all feature names are present and consistent across data structures

        self._encoders = encoders

        self._mlps = mlps

    # Encoding (feature extraction)
    def encode(self, inputs, dataset_name, lengths=None):
        # if self._encoders[dataset_name] == 'no_op':
        #     return inputs
        return self._encoders[dataset_name](inputs, lengths)

    def get_predictions(self, batch, dataset_name, is_training):
        # Returns most likely label given conditioning variables (only run this on eval data)
        inputs = batch[self._hps.input_key]
        input_lengths = batch[self._hps.token_lengths_key]

        features = self.encode(inputs, dataset_name, lengths=input_lengths)

        logits = self._mlps[dataset_name](features, is_training)

        res = tf.argmax(logits, axis=1)
        # res = tf.expand_dims(res, axis=1)

        return res

    def get_loss(self, batch, dataset_name, features=None, is_training=True):
        # Returns most likely label given conditioning variables (only run this on eval data)
        inputs = batch[self._hps.input_key]
        input_lengths = batch[self._hps.token_lengths_key]
        labels = batch[self._hps.label_key]

        if features is None:
            features = self.encode(inputs, dataset_name, lengths=input_lengths)

        logits = self._mlps[dataset_name](features, is_training)

        # loss
        ce = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=tf.cast(labels, dtype=tf.int32)))

        # l2 regularization
        variables = tf.trainable_variables()
        l2_weight_penalty = tf.add_n([tf.nn.l2_loss(v) for v in variables
                                      if 'bias' not in v.name]) * self._hps.l2_weight
        loss = tf.reduce_mean(ce) + l2_weight_penalty

        return loss

    def get_multi_task_loss(self, dataset_batches, is_training):
        # dataset_batches: map from dataset names to training batches (one batch per dataset)
        # we assume only one dataset's labels are observed; the rest are unobserved

        losses = dict()
        total_loss = 0.0
        assert len(dataset_batches) == len(self._hps.alphas)
        assert sum(self._hps.alphas) == 1.0
        for dataset_batch, alpha in zip(dataset_batches.items(), self._hps.alphas):
            # We assume that the encoders and decoders always use the same fields/features
            # (given by the keys in the batch accesses below)
            dataset_name, batch = dataset_batch
            loss = self.get_loss(batch=batch, dataset_name=dataset_name, features=None, is_training=is_training)
            total_loss += alpha * loss
            losses[dataset_name] = loss

        return total_loss

    @property
    def encoders(self):
        return self._encoders

    @property
    def hp(self):
        return self._hps

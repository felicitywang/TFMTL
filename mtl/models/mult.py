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

from mtl.util.common import preoutput_MLP
from mtl.util.layers import dense_layer

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
                 hps=None,
                 is_training=None):

        # class_sizes: map from feature names to cardinality of label sets
        # dataset_order: list of features in some fixed order
        #   (concatenated order matters for decoding)
        # encoders: one per dataset

        assert class_sizes is not None
        assert dataset_order is not None
        assert encoders is not None
        assert hps is not None
        assert is_training is not None

        self._hp = hps

        self._class_sizes = class_sizes

        self._dataset_order = dataset_order

        assert set(class_sizes.keys()) == set(
            dataset_order)  # all feature names are present and consistent across data structures

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

        self._mlp = dict()
        for k, v in class_sizes.items():
            self._mlp[k] = tf.make_template('py_{}'.format(k),
                                            mlp,
                                            output_size=v,
                                            embed_dim=self._hp.embed_dim,
                                            num_layers=self._hp.num_layers,
                                            activation=tf.nn.relu,
                                            dropout_rate=self._hp.dropout_rate,
                                            is_training=is_training
                                            )

    # Encoding (feature extraction)
    def encode(self, inputs, feature_name, lengths=None):
        if self._encoders[feature_name] == 'no_op':
            return inputs
        return self._encoders[feature_name](inputs, lengths)

    def get_predictions(self, batch, feature_name, features=None):
        # Returns most likely label given conditioning variables (only run this on eval data)
        inputs = batch[self._hp.input_key]
        input_lengths = batch[self._hp.token_lengths_key]

        if features is None:
            features = self.encode(inputs, feature_name, lengths=input_lengths)

        logits = self._mlp[feature_name](features)

        res = tf.argmax(logits, axis=1)
        # res = tf.expand_dims(res, axis=1)

        return res

    def get_loss(self, batch, feature_name, features=None):
        # Returns most likely label given conditioning variables (only run this on eval data)
        inputs = batch[self._hp.input_key]
        input_lengths = batch[self._hp.token_lengths_key]
        labels = batch[self._hp.label_key]

        if features is None:
            features = self.encode(inputs, feature_name, lengths=input_lengths)

        logits = self._mlp[feature_name](features)

        # loss
        ce = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=tf.cast(labels, dtype=tf.int32)))

        # l2 regularization
        variables = tf.trainable_variables()
        l2_weight_penalty = tf.add_n([tf.nn.l2_loss(v) for v in variables
                                      if 'bias' not in v.name]) * self._hp.l2_weight
        loss = tf.reduce_mean(ce) + l2_weight_penalty

        return loss

    def get_multi_task_loss(self, dataset_batches):
        # dataset_batches: map from dataset names to training batches (one batch per dataset)
        # we assume only one dataset's labels are observed; the rest are unobserved

        losses = dict()
        total_loss = 0.0
        assert len(dataset_batches) == len(self._hp.alphas)
        assert sum(self._hp.alphas) == 1.0
        for dataset_batch, alpha in zip(dataset_batches.items(), self._hp.alphas):
            # We assume that the encoders and decoders always use the same fields/features
            # (given by the keys in the batch accesses below)
            dataset_name, batch = dataset_batch
            loss = self.get_loss(batch=batch, feature_name=dataset_name, features=None)
            total_loss += alpha * loss
            losses[dataset_name] = loss

        return total_loss

    @property
    def encoders(self):
        return self._encoders

    @property
    def hp(self):
        return self._hp


def mlp(inputs, output_size, embed_dim, num_layers=2, activation=tf.nn.elu, dropout_rate=0.5, is_training=True):
    # Returns logits (unnormalized log probabilities)
    x = preoutput_MLP(inputs, embed_dim, num_layers=num_layers, activation=activation)
    # x = tf.layers.dropout(x, rate=dropout_rate,
    #                       training=is_training)
    x = dense_layer(x, output_size, 'logits', activation=None)
    return x

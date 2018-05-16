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
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import tensorflow as tf

from mtl.layers.mlp import dense_layer, mlp
from mtl.util.encoder_factory import build_encoders

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
               # TODO keep one of hps and args only
               hps=None,
               args=None):

    # class_sizes: map from feature names to cardinality of label sets
    # dataset_order: list of features in some fixed order
    #   (concatenated order matters for decoding)
    # encoders: one per dataset

    assert class_sizes is not None
    assert dataset_order is not None
    assert hps is not None
    assert args is not None

    self._class_sizes = class_sizes
    self._dataset_order = dataset_order
    self._hps = hps
    # self._args = args

    # all feature names are present and consistent across data structures
    assert set(class_sizes.keys()) == set(dataset_order)

    self._encoders = build_encoders(args)

    if self._hps.experiment_name != "RUDER_NAACL_18":
      self._mlps_shared = build_mlps(hps, is_shared=True)
    self._mlps_private = build_mlps(hps, is_shared=False)

    self._logits = build_logits(class_sizes, self._hps.experiment_name)

  # Encoding (feature extraction)
  def encode(self,
             inputs,
             dataset_name,
             # is_training,
             lengths=None,
             additional_extractor_kwargs=dict()):
    # Also apply arguments that aren't `inputs` or `lengths`
    # (such as `indices` for the serial bi-RNN extractor)
    return self._encoders[dataset_name](inputs,
                                        lengths,
                                        # is_training,
                                        **additional_extractor_kwargs[
                                          dataset_name])

  def get_predictions(self,
                      batch,
                      batch_source,
                      dataset_name,
                      additional_extractor_kwargs=dict()):
    # Returns most likely label given conditioning variables (only
    # run this on eval data)

    if self._hps.experiment_name == "RUDER_NAACL_18":
      assert batch_source == dataset_name

    for dataset, dataset_path in zip(self._hps.datasets,
                                     self._hps.dataset_paths):
      if dataset == batch_source:
        with open(os.path.join(dataset_path, 'args.json')) as f:
          text_field_names = json.load(f)['text_field_names']
          if self._hps.experiment_name == "RUDER_NAACL_18":
            # print("text_field_names={}".format(text_field_names))
            assert text_field_names == ['seq1', 'seq2']

    x = list()
    input_lengths = list()
    for text_field_name in text_field_names:
      input_lengths.append(batch[text_field_name + '_length'])
      if self._hps.input_key == 'tokens':
        x.append(batch[text_field_name])
      elif self._hps.input_key == 'bow':
        x.append(batch[text_field_name + '_bow'])
      elif self._hps.input_key == 'tfidf':
        x.append(batch[text_field_name + '_tfidf'])
      else:
        raise ValueError("unrecognized input key: %s" % (self._hps.input_key))

    # Turn back into single value instead of list
    if len(x) == 1:
      x = x[0]
    if len(input_lengths) == 1:
      input_lengths = input_lengths[0]

    x = self.encode(x,
                    dataset_name,
                    # is_training=False,
                    lengths=input_lengths,
                    additional_extractor_kwargs=additional_extractor_kwargs)

    if self._hps.experiment_name != "RUDER_NAACL_18":
      x = self._mlps_shared[dataset_name](x, is_training=False)
    x = self._mlps_private[dataset_name](x, is_training=False)

    x = self._logits[dataset_name](x)

    res = tf.argmax(x, axis=1)
    # res = tf.expand_dims(res, axis=1)

    return res

  def get_loss(self,
               batch,
               batch_source,  # which dataset the batch is from
               dataset_name,  # we predict labels w.r.t. this dataset
               additional_extractor_kwargs=dict(),
               features=None,
               is_training=True):
    # Returns most likely label given conditioning variables (only
    # run this on eval data)

    if self._hps.experiment_name == "RUDER_NAACL_18":
      assert batch_source == dataset_name

    for dataset, dataset_path in zip(self._hps.datasets,
                                     self._hps.dataset_paths):
      if dataset == batch_source:
        with open(os.path.join(dataset_path, 'args.json')) as f:
          text_field_names = json.load(f)['text_field_names']
          if self._hps.experiment_name == "RUDER_NAACL_18":
            # print("text_field_names={}".format(text_field_names))
            assert text_field_names == ['seq1', 'seq2']

    x = list()
    input_lengths = list()
    for text_field_name in text_field_names:
      input_lengths.append(batch[text_field_name + '_length'])
      if self._hps.input_key == 'tokens':
        x.append(batch[text_field_name])
      elif self._hps.input_key == 'bow':
        x.append(batch[text_field_name + '_bow'])
      elif self._hps.input_key == 'tfidf':
        x.append(batch[text_field_name + '_tfidf'])
      else:
        raise ValueError("unrecognized input key: %s" % (self._hps.input_key))

    # Turn back into single value instead of list
    if len(x) == 1:
      x = x[0]
    if len(input_lengths) == 1:
      input_lengths = input_lengths[0]

    labels = batch[self._hps.label_key]

    # TODO remove this?
    if features is None:
      x = self.encode(x,
                      dataset_name,
                      # is_training=is_training,
                      lengths=input_lengths,
                      additional_extractor_kwargs=additional_extractor_kwargs)
    else:
      x = features

    if self._hps.experiment_name != "RUDER_NAACL_18":
      x = self._mlps_shared[dataset_name](x, is_training=is_training)
    x = self._mlps_private[dataset_name](x, is_training=is_training)

    x = self._logits[dataset_name](x)

    # loss
    #ce = tf.reduce_mean(
    #  tf.nn.sparse_softmax_cross_entropy_with_logits(
    #    logits=x, labels=tf.cast(labels, dtype=tf.int32)))
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                        labels=tf.cast(labels, dtype=tf.int32))
    
    loss = tf.reduce_mean(ce)  # average loss per training example

    return loss

  def get_multi_task_loss(self,
                          dataset_batches,
                          is_training,
                          additional_extractor_kwargs=dict()):
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
      loss = self.get_loss(batch=batch,
                           batch_source=dataset_name,
                           dataset_name=dataset_name,
                           additional_extractor_kwargs=
                           additional_extractor_kwargs,
                           features=None,
                           is_training=is_training)
      total_loss += alpha * loss
      losses[dataset_name] = loss

    # l2 regularization
    variables = tf.trainable_variables()
    l2_weight_penalty = tf.add_n([
      tf.nn.l2_loss(v) for v in variables if 'bias' not in v.name
    ]) * self._hps.l2_weight

    total_loss += l2_weight_penalty

    # return total_loss
    return losses

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
                                  activation=tf.tanh,
                                  # input_keep_prob=hps.input_keep_prob,
                                  input_keep_prob=1,
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
        activation=tf.tanh,
        # TODO args.dropout_rate to keep_prob
        input_keep_prob=1,
        # TODO ?
        batch_normalization=False,
        # TODO ?
        layer_normalization=False,
        # output_keep_prob=hps.output_keep_prob,
        output_keep_prob=1,
      )

  return mlps


def build_logits(class_sizes,
                 experiment_name):
  if experiment_name == "RUDER_NAACL_18":
    activation = tf.tanh
  else:
    activation = None
  logits = dict()
  for k, v in class_sizes.items():
    logits[k] = tf.make_template('logit_{}'.format(k),
                                 dense_layer,
                                 name='logits',
                                 output_size=v,
                                 activation=activation
                                 )
  return logits

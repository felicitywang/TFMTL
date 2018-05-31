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
from mtl.util.constants import EXP_NAMES

logging = tf.logging
eps = 1e-5


class Mult(object):
  def __init__(self,
               class_sizes,
               dataset_order,
               hps):

    # class_sizes: map from feature names/dataset names to cardinality of label sets
    # dataset_order: list of features/datasets in some fixed order
    #   (concatenated order could matter for decoding)

    # all feature names are present and consistent across data structures
    if set(class_sizes.keys()) != set(dataset_order):
      raise ValueError("class_sizes must have same elements as dataset_order")
    
    self._class_sizes   = class_sizes
    self._dataset_order = dataset_order
    self._hps           = hps

    self._encoders     = build_encoders(self._hps)  # encoders: one per dataset
    self._mlps_shared  = build_mlps(hps, is_shared=True)
    self._mlps_private = build_mlps(hps, is_shared=False)
    self._logit_layers = build_logit_layers(class_sizes, self._hps.experiment_name)


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


  def get_logits(self,
                 batch,
                 batch_source,  # name of dataset that batch is from
                 dataset_name,  # name of dataset whose labels we are encoding/decoding/predicting wrt
                 is_training,
                 additional_extractor_kwargs=dict()):
    if self._hps.experiment_name in [EMNLP_18]:
      if batch_source != dataset_name:
        raise ValueError("The batch and label set must come from the same dataset (exp=%s)" % (self._hps.experiment_name))

    if batch_source not in self._hps.datasets:
      raise ValueError("Unrecognized batch source=%s" % (batch_source))

    batch_source_idx = self._hps.datasets.index(batch_source)
    batch_source_dataset_path = self._hps.dataset_paths[batch_source_idx]  # assumes same ordering of datasets in hps.datasets and hps.dataset_paths
    with open(os.path.join(batch_source_dataset_path, 'args.json')) as f:
      text_field_names = json.load(f)['text_field_names']
      if self._hps.experiment_name in [RUDER_NAACL_18, EMNLP_18]:
        if text_field_names != ['seq1', 'seq2']:
          raise ValueError("Text field names must be [seq1, seq2] (exp=%s)" % (self._hps.experiment_name))
      else:
        # Requirements/constraints on text_field_names (for other experiments) go here
        pass

    x = list()
    input_lengths = list()
    for text_field_name in text_field_names:
      input_lengths.append(batch[text_field_name + '_length'])  # TODO: un-hard-code this
      if self._hps.input_key == 'tokens':
        x.append(batch[text_field_name])
      elif self._hps.input_key == 'bow':
        x.append(batch[text_field_name + '_bow'])
      elif self._hps.input_key == 'tfidf':
        x.append(batch[text_field_name + '_tfidf'])
      else:
        raise ValueError("unrecognized input key: %s" % (self._hps.input_key))

    x = self.encode(x,
                    dataset_name,
                    lengths=input_lengths,
                    additional_extractor_kwargs=additional_extractor_kwargs)
    x = self._mlps_shared[dataset_name](x, is_training=is_training)
    x = self._mlps_private[dataset_name](x, is_training=is_training)
    x = self._logit_layers[dataset_name](x)

    return x


  def get_predictions(self,
                      batch,
                      batch_source,
                      dataset_name,
                      additional_extractor_kwargs=dict()):
    # Returns most likely label given conditioning variables
    # (run this on eval data ONLY)

    x = self.get_logits(batch,
                        batch_source,
                        dataset_name,
                        is_training=False,
                        additional_extractor_kwargs=additional_extractor_kwargs)
    res = tf.argmax(x, axis=1)
    # res = tf.expand_dims(res, axis=1)

    return res


  def get_loss(self,
               batch,
               batch_source,  # which dataset the batch is from
               dataset_name,  # we predict labels w.r.t. this dataset
               additional_extractor_kwargs=dict(),
               is_training=True):

    # returns a scalar loss for each batch
    
    x = self.get_logits(batch,
                        batch_source,
                        dataset_name,
                        is_training=is_training,
                        additional_extractor_kwargs=additional_extractor_kwargs)
    labels = batch[self._hps.label_key]
    
    # loss
    softmax_xent = tf.nn.sparse_softmax_cross_entropy_with_logits
    ce = softmax_xent(logits=x, labels=tf.cast(labels, dtype=tf.int32))

    loss = tf.reduce_mean(ce)  # average loss per training example

    return loss


  def get_l2_penalty(self):
    if self._hps.l2_weight < 0.0:
      raise ValueError("L2 weight must be non-negative")

    if self._hps.l2_weight == 0.0:
      l2_weight_penalty = 0.0
    else:
      tvars = tf.trainable_variables()
      l2_weight_penalty = tf.add_n([tf.nn.l2_loss(v) for v in tvars
                                    if 'bias' not in v.name])
      l2_weight_penalty *= self._hps.l2_weight

    return l2_weight_penalty

  
  def get_multi_task_loss(self,
                          dataset_batches,
                          is_training,
                          additional_extractor_kwargs=dict()):
    # dataset_batches: map from dataset names to training batches
    # (one batch per dataset)
    #
    # we assume only one dataset's labels
    #   are observed at a time; the rest are unobserved

    if len(dataset_batches) != len(self._hps.alphas):
      raise ValueError("The calculation of multi-task loss requires the same number of batches as alpha values")
    if abs(sum(self._hps.alphas) - 1.0) > eps:
      raise ValueError("The alpha values must sum to 1")

    losses = dict()
    total_loss = 0.0
    for dataset_batch, alpha in zip(dataset_batches.items(),
                                    self._hps.alphas):
      # We assume that the encoders and decoders always use the
      # same fields/features (given by the keys in the batch
      # accesses below)
      batch_source, batch = dataset_batch
      if self._hps.experiment_name in [EMNLP_18]:
        # encode/decode wrt same dataset that batch came from        
        dataset_name = batch_source
        additional_extractor_kwargs = additional_extractor_kwargs
      else:
        raise NotImplementedError("Must specify wrt which dataset to get loss")
      loss = self.get_loss(batch=batch,
                           batch_source=batch_source,
                           dataset_name=dataset_name,
                           additional_extractor_kwargs=
                           additional_extractor_kwargs,
                           is_training=is_training)
      total_loss += alpha * loss
      losses[dataset_name] = loss

    # l2 regularization
    l2_weight_penalty = self.get_l2_penalty()

    total_loss += l2_weight_penalty

    if self._hps.experiment_name in [RUDER_NAACL_18, EMNLP_18]:
      return losses
    else:
      return total_loss


def build_mlps(hps, is_shared):
  mlps = dict()
  if is_shared:
    # One shared MLP for all features/datasets
    mlp_shared = tf.make_template('mlp_shared',
                                  mlp,
                                  hidden_dims=hps.shared_hidden_dims,
                                  num_layers=hps.shared_mlp_layers,
                                  activation=tf.tanh,
                                  input_keep_prob=hps.input_keep_prob,
                                  batch_normalization=hps.batch_normalization,
                                  layer_normalization=hps.layer_normalization,
                                  output_keep_prob=hps.output_keep_prob)
    for dataset in hps.datasets:
      mlps[dataset] = mlp_shared

  else:
    # One MLP for each feature/dataset
    for dataset in hps.datasets:
      mlps[dataset] = tf.make_template(
        'mlp_{}'.format(dataset),
        mlp,
        hidden_dims=hps.private_hidden_dims,
        num_layers=hps.private_mlp_layers,
        activation=tf.tanh,
        input_keep_prob=hps.input_keep_prob,
        batch_normalization=hps.batch_normalization,
        layer_normalization=hps.layer_normalization,
        output_keep_prob=hps.output_keep_prob)

  return mlps


def build_logit_layers(class_sizes,
                       experiment_name):
  if experiment_name in [RUDER_NAACL_18, EMNLP_18]:
    activation = tf.tanh
  else:
    activation = None

  # task-specific logits output layer
  logits = dict()
  for k, v in class_sizes.items():
    logits[k] = tf.make_template('logit_{}'.format(k),
                                 dense_layer,
                                 name='logits',
                                 output_size=v,
                                 activation=activation)
  return logits

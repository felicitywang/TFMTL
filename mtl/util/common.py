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
from six.moves import xrange

from mtl.util.layers import dense_layer


# General helpers
def listify(x):
  # Convert inputs into a list if it is not a list already
  if type(x) is not list:
    return [x]
  else:
    return x


def maybe_concat(x):
  # Concatenate inputs if it is a list (of Tensors)
  # Provides a uniform way of conditioning on one variable vs multiple variables
  if type(x) is list:
    return tf.concat(x, axis=1)
  else:
    return x


# Distributions
def preoutput_MLP(inputs, embed_dim, num_layers=2, activation=tf.nn.elu):
  # Returns output of last layer of N-layer dense MLP that can then be
  # passed to an output layer
  x = maybe_concat(inputs)
  for i in xrange(num_layers):
    x = dense_layer(x, embed_dim, 'l{}'.format(i+1), activation=activation)
  return x


def MLP_gaussian_posterior(inputs, embed_dim, latent_dim, min_var=0.0):
  # Returns mean and variance parametrizing a (multivariate) Gaussian
  x = preoutput_MLP(inputs, embed_dim, num_layers=2, activation=tf.nn.elu)
  zm = dense_layer(x, latent_dim, 'zm', activation=None)
  zv = dense_layer(x, latent_dim, 'zv', tf.nn.softplus)  # variance must be positive
  if min_var > 0.0:
    zv = tf.maximum(min_var, zv)  # ensure zv is *no smaller* than min_var
  return zm, zv


def MLP_unnormalized_log_categorical(inputs, output_size, embed_dim):
  # Returns logits (unnormalized log probabilities)
  x = preoutput_MLP(inputs, embed_dim, num_layers=2, activation=tf.nn.elu)
  x = dense_layer(x, output_size, 'logit', activation=None)
  return x


def MLP_ordinal(inputs, embed_dim):
  # Returns scalar output
  x = preoutput_MLP(inputs, embed_dim, num_layers=2, activation=tf.nn.elu)
  x = dense_layer(x, 1, 'val', activation=None)
  return x

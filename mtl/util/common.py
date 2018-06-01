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

from mtl.layers import dense_layer


def listify(x):
    if not isinstance(x, (list, tuple)):
        return [x]
    else:
        return x


def maybe_concat(x):
    if type(x) is list:
        return tf.concat(x, axis=1)
    else:
        return x


def preoutput_MLP(inputs, hidden_dim=512, num_layers=2, activation=tf.nn.selu):
    assert type(inputs) is list or type(inputs) is tf.Tensor, type(inputs)
    x = maybe_concat(inputs)
    for i in xrange(num_layers):
        x = dense_layer(x, hidden_dim, 'l{}'.format(i + 1),
                        activation=activation)
    return x


def MLP_gaussian_posterior(inputs, latent_dim, **kwargs):
    x = preoutput_MLP(inputs, **kwargs)
    zm = dense_layer(x, latent_dim, 'zm', activation=None)
    zv = dense_layer(x, latent_dim, 'zv', tf.nn.softplus)
    return zm, zv


def MLP_unnormalized_log_categorical(inputs, output_size, **kwargs):
    return dense_layer(preoutput_MLP(inputs, **kwargs), output_size,
                       'logits', activation=None)


def MLP_ordinal(inputs, **kwargs):
    return dense_layer(preoutput_MLP(inputs, **kwargs), 1, 'output',
                       activation=None)


def validate_extractor_inputs(*inputs):
  it = iter(inputs)
  num_stages = len(next(it))
  if not all(len(l) == num_stages for l in it):
    raise ValueError("all list arguments must have the same length")

  if num_stages <= 0:
    raise ValueError("must specify arguments for at least \
                      one stage of extractor")

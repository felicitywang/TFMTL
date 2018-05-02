# Copyright 2018 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.seq2seq import sequence_loss
from tensorflow.contrib.training import HParams
from mtl.layers.t2t import conv_block_wn
from mtl.util.hparams import get_activation_fn
import mtl.util.registry as registry


@registry.register_hparams
def resnet_default():
  hps = tf.contrib.training.HParams(
    kernel_height = 3,
    kernel_width = 1,
    num_hidden_layer = 3,
    num_conv_group = 1,
    nonlinearity = 'relu',
    keep_prob = 1.0
  )
  return hps


@registry.register_decoder
def resnet(x, global_conditioning=None,
           hp=resnet_default(), name='resnet'):
  ndims = len(x.get_shape().as_list())
  if ndims != 4:
    raise ValueError("expected 4D input, got %dD" % ndims)
  hidden_dim = x.get_shape().as_list()[-1]
  hp = hp()
  if hp.num_conv_group < 1:
    raise ValueError('num_conv_group < 1')
  nonlinearity = get_activation_fn(hp.nonlinearity)
  conv_block_fn = conv_block_wn
  with tf.variable_scope(name):
    k = (hp.kernel_height, hp.kernel_width)
    dilations_and_kernels = [((2**i, 1), k)
                             for i in xrange(hp.num_hidden_layer)]
    for i in xrange(hp.num_conv_group):
      with tf.variable_scope("repeat_%d" % i):
        y = conv_block_fn(x,
                          hidden_dim,
                          dilations_and_kernels,
                          padding="LEFT",
                          nonlinearity=nonlinearity,
                          name="residual_conv_%d" % i)
        if hp.nonlinearity is tf.nn.selu:
          y = tf.contrib.nn.alpha_dropout(y, hp.keep_prob)
        else:
          y = tf.nn.dropout(y, hp.keep_prob)
        x += y
    return x

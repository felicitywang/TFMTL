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
from six.moves import xrange
from mtl.layers.t2t import conv_block_wn
from mtl.util.hparams import get_activation_fn
import mtl.util.registry as registry


@registry.register_hparams
def resnet_default():
    hps = tf.contrib.training.HParams(
        kernel_height=3,
        kernel_width=1,
        num_hidden_layer=4,
        num_conv_group=2,
        nonlinearity='relu',
        keep_prob=0.75
    )
    return hps


@registry.register_hparams
def resnet_single_layer():
    hps = tf.contrib.training.HParams(
        kernel_height=3,
        kernel_width=1,
        num_hidden_layer=3,
        num_conv_group=1,
        nonlinearity='relu',
        keep_prob=0.75
    )
    return hps


@registry.register_hparams
def resnet_large():
    hps = tf.contrib.training.HParams(
        kernel_height=3,
        kernel_width=1,
        num_hidden_layer=5,
        num_conv_group=3,
        nonlinearity='relu',
        keep_prob=0.5
    )
    return hps


@registry.register_hparams
def resnet_huge():
    hps = tf.contrib.training.HParams(
        kernel_height=3,
        kernel_width=1,
        num_hidden_layer=5,
        num_conv_group=5,
        nonlinearity='relu',
        keep_prob=0.5
    )
    return hps


@registry.register_decoder
def resnet(x, is_training, global_conditioning=None,
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

    if global_conditioning is not None:
        with tf.variable_scope("global_cond_proj"):
            h = tf.layers.dense(global_conditioning, hidden_dim, use_bias=True)
    else:
        h = None

    with tf.variable_scope(name):
        assert hp.kernel_height is not None
        assert hp.kernel_width is not None
        k = (hp.kernel_height, hp.kernel_width)
        dilations_and_kernels = [((2 ** i, 1), k)
                                 for i in xrange(hp.num_hidden_layer)]
        for i in xrange(hp.num_conv_group):
            with tf.variable_scope("repeat_%d" % i):
                y = conv_block_fn(x,
                                  hidden_dim,
                                  dilations_and_kernels,
                                  padding="LEFT",
                                  nonlinearity=nonlinearity,
                                  global_conditioning=h,
                                  name="residual_conv_%d" % i)
                if is_training is True:
                    if hp.nonlinearity is tf.nn.selu:
                        y = tf.contrib.nn.alpha_dropout(y, hp.keep_prob)
                    else:
                        y = tf.nn.dropout(y, hp.keep_prob)
                x += y

        assert x.get_shape().as_list()[2] == 1
        return x

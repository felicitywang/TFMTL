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


def get_activation_fn(s):
    if s == 'elu':
        act_fn = tf.nn.elu
    elif s == 'selu':
        act_fn = tf.nn.selu
    elif s == 'relu':
        act_fn = tf.nn.relu
    elif s == 'none' or None:
        act_fn = None
    elif s == 'tanh':
        act_fn = tf.nn.tanh
#    elif s == 'prelu':
#        act_fn = prelu
    else:
        raise ValueError("unsupported activation fn: %s" % s)
    return act_fn


def dict2func(d):
  """Converts all strings in a dictionary to their
  corresponding functions (if applicable)"""
  res = dict()
  for k, v in d.items():
    if isinstance(v, dict):
      res[k] = dict2func(v)
    else:
      res[k] = str2func(v)
  return res


def str2func(s):
  from mtl.util.embed import (embed_sequence,
                              no_op_embedding)

  from mtl.encoders.paragram import paragram_phrase
  from mtl.encoders.cnn import conv_and_pool
  from mtl.encoders.rnn import rnn_and_pool
  from mtl.encoders.no_op import no_op_encoding

  from mtl.util.reducers import (reduce_avg_over_time,
                                 reduce_var_over_time,
                                 reduce_max_over_time,
                                 reduce_min_over_time)

  functions = {
    "embed_sequence": embed_sequence,
    "no_op_embedding": no_op_embedding,

    "paragram": paragram_phrase,
    "conv_and_pool": conv_and_pool,
    "rnn_and_pool": rnn_and_pool,
    "no_op_encoding": no_op_encoding,

    "reduce_min_over_time": reduce_min_over_time,
    "reduce_max_over_time": reduce_max_over_time,
    "reduce_avg_over_time": reduce_avg_over_time,
    "reduce_var_over_time": reduce_var_over_time,

    "tf.nn.relu": tf.nn.relu,
    "tf.nn.elu": tf.nn.elu,
  }

  res = functions[s] if s in functions else s
  return res


def update_hparams_from_args(hps, args, log=True):
    """
    Updates hps with values of matching keys in args

    Inputs:
      hps: HParams object
      args: argparse Namespace returned from parse_args()
    """
    if log:
        tf.logging.info("Updating hyperparameters from command line arguments")
    opts = vars(args)
    for hps_k, hps_v in hps.values().items():
        if hps_k in opts:
            if log:
                tf.logging.info("  %20s: %10s -> %10s",
                                hps_k,
                                hps_v,
                                opts[hps_k])
            hps.set_hparam(hps_k, opts[hps_k])

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
  elif s == 'prelu':
    act_fn = prelu
  else:
    raise ValueError("unsupported activation fn: %s" % s)
  return act_fn


def update_hparams_from_args(hps, args, log=True):
  """
  Updates hps with values of matching keys in args

  Inputs:
    hps: HParams object
    args: argparse Namespace returned from parse_args()
  """
  if log:
    tf.logging.info("Updating hyper-parameters from command-line arguments.")
  opts = vars(args)
  for hps_k, hps_v in hps.values().items():
    if hps_k in opts:
      if log:
        tf.logging.info("  %20s: %10s -> %10s", hps_k, hps_v, opts[hps_k])
      hps.set_hparam(hps_k, opts[hps_k])

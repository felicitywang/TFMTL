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


def normalize_logits(logits, dims=None):
  assert len(logits.get_shape()) == 2
  logits -= tf.reduce_logsumexp(logits, axis=1, keepdims=True)
  if dims is None:
    return logits
  else:
    batch_size = tf.shape(logits)[0]
    return tf.reshape(logits, [batch_size] + dims)


def marginal_log_prob(normalized_logits, target_index):
  ndims = len(normalized_logits.get_shape())
  reduce_axis = list(xrange(ndims))
  del reduce_axis[target_index+1]
  del reduce_axis[0]
  return tf.reduce_logsumexp(normalized_logits, reduce_axis)


def conditional_log_prob(normalized_logits, target_index, cond_index,
                         logits=None, normalize=False):
  assert target_index != cond_index
  ndims = len(normalized_logits.get_shape())
  ln_p_cond = marginal_log_prob(normalized_logits, cond_index)
  reduce_axis = list(xrange(ndims))
  reduce_axis.remove(target_index + 1)
  reduce_axis.remove(cond_index + 1)
  reduce_axis.remove(0)
  marginal_ln_joint = tf.reduce_logsumexp(normalized_logits,
                                          reduce_axis)
  if cond_index > target_index:
    marginal_ln_joint = tf.transpose(marginal_ln_joint,
                                     perm=[0, 2, 1])
  ln_p_cond = tf.expand_dims(ln_p_cond, axis=-1)
  final_dim = tf.shape(marginal_ln_joint)[-1]
  ln_p_cond = tf.tile(ln_p_cond, [1, 1, final_dim])
  return marginal_ln_joint - ln_p_cond

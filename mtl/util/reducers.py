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


def reduce_min_over_time(x, lengths=None, time_axis=1):
    if lengths is None:
        return tf.reduce_min(x, keepdims=False, axis=time_axis)
    else:
        raise ValueError("not supported")


def reduce_max_over_time(x, lengths=None, time_axis=1):
    return tf.reduce_max(x, keepdims=False, axis=time_axis)


def reduce_avg_over_time(x, lengths=None, time_axis=1):
    if lengths is None:
        return tf.reduce_mean(x, keepdims=False, axis=time_axis)
    s = tf.reduce_sum(x, axis=time_axis)

    rank_s = len(s.get_shape().as_list())
    rank_l = len(lengths.get_shape().as_list())

    if rank_l == rank_s - 1:
        lengths = tf.expand_dims(lengths, 1)

    return tf.divide(s, tf.to_float(lengths))


def reduce_var_over_time(x, lengths=None, avg=None, time_axis=1):
    time_dim = x.get_shape()[time_axis]
    if avg is None:
        avg = reduce_avg_over_time(x, lengths=lengths, time_axis=time_axis)
    avg = tf.expand_dims(avg, axis=time_axis)
    avg = tf.tile(avg, [1, time_dim])
    sd = tf.squared_difference(x, avg)
    if lengths is None:
        return reduce_avg_over_time(sd, lengths=None, time_axis=time_axis)
    else:
        mask = tf.to_float(tf.sequence_mask(lengths))
        masked_sd = tf.multiply(sd, mask)
        return reduce_avg_over_time(masked_sd, lengths=lengths,
                                    time_axis=time_axis)


def reduce_over_time(x, lengths=None, max=True, min=False, avg=True,
                     var=False, time_axis=1):
    summaries = []
    if min:
        summaries += [reduce_min_over_time(x,
                                           lengths=lengths,
                                           time_axis=time_axis)]
    if max:
        summaries += [reduce_max_over_time(x,
                                           lengths=lengths,
                                           time_axis=time_axis)]
    if avg:
        summaries += [reduce_avg_over_time(x,
                                           lengths=lengths,
                                           time_axis=time_axis)]
    if var:
        summaries += [reduce_var_over_time(x,
                                           lengths=lengths,
                                           time_axis=time_axis)]
    return tf.concat(summaries, axis=1)

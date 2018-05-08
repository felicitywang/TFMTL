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

from collections import namedtuple

import tensorflow as tf
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.ops import parsing_ops


class Pipeline(object):
  def __init__(self, tfrecord_file, feature_map, batch_size=32,
               num_threads=4, prefetch_buffer_size=1,
               static_max_length=None, shuffle_buffer_size=10000,
               shuffle=True, num_epochs=None, one_shot=False):
    self._feature_map = feature_map
    self._batch_size = batch_size
    self._static_max_length = static_max_length

    # Initialize the dataset
    dataset = tf.data.TFRecordDataset(tfrecord_file)

    # Maybe randomize
    if shuffle:
      dataset = dataset.shuffle(shuffle_buffer_size)

    # Maybe repeat
    if num_epochs is None:
      dataset = dataset.repeat()  # repeat indefinitely
    elif num_epochs > 1:
      dataset = dataset.repeat(count=num_epochs)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(self.parse_example,
                          num_parallel_calls=num_threads)

    # Pre-fetch a batch for faster processing
    dataset = dataset.prefetch(prefetch_buffer_size)

    # Get the iterator
    if one_shot:
      self._iterator = dataset.make_one_shot_iterator()
    else:
      self._iterator = dataset.make_initializable_iterator()
      self._init_op = self._iterator.initializer

    # Get outputs
    self._outputs = self._iterator.get_next()

    # Map to features
    index = 0
    result = {}
    for key in sorted(self._feature_map.keys()):
      result[key] = self._outputs[index]
      index += 1
    self._result = result

  def pad(self, t):
    s = tf.shape(t)
    paddings = [[0, 0], [0, self._static_max_length-s[1]]]
    x = tf.pad(t, paddings, 'CONSTANT', constant_values=0)
    x = tf.reshape(x, [s[0], self._static_max_length])
    assert x.get_shape().as_list()[1] is self._static_max_length
    return x

  def parse_example(self, serialized):
    parsed = parsing_ops.parse_example(serialized, self._feature_map)
    result = []
    for key in sorted(self._feature_map.keys()):
      val = parsed[key]
      if isinstance(val, sparse_tensor_lib.SparseTensor):
        dense_tensor = tf.sparse_tensor_to_dense(val)
        if self._static_max_length is not None:
          dense_tensor = self.pad(dense_tensor)
        result.append(dense_tensor)
      else:
        result.append(val)
    return tuple(result)

  @property
  def iterator(self):
    return self._iterator

  @property
  def init_op(self):
    return self._init_op

  @property
  def batch(self):
    return self._result


# namedtuple for bucket_info object (used in Pipeline)
# func: a mapping from examples to tf.int64 keys
# pads: a set of tf shapes that correspond to padded examples
bucket_info = namedtuple("bucket_info", "func pads")


def int64_feature(value):
  """ Takes a single int (e.g. 3) and converts it to a tf Feature """
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(sequence):
  """ Sequence of ints (e.g [1,2,3]) to TF feature """
  return tf.train.Feature(int64_list=tf.train.Int64List(value=sequence))

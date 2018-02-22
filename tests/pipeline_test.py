#! /usr/bin/env python

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

import os

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from mtl.util.pipeline import Pipeline
from mtl.util.pipeline import bucket_info
from mtl.util.pipeline import int64_feature
from mtl.util.pipeline import int64_list_feature


def random_sequences(N, maxlen, maxint):
  def random_length():
    return np.random.randint(1, maxlen + 1)

  def randseq():
    return np.random.randint(low=1,
                             high=maxint,
                             size=random_length()).tolist()

  for _ in range(N):
    yield randseq()


class PipelineTests(tf.test.TestCase):
  def setUp(self):
    self._N = 16
    self._batch_size = 4

  def write_examples(self):
    tmp_dir = self.get_temp_dir()
    file_name = os.path.join(tmp_dir, 'records.tf')
    with tf.python_io.TFRecordWriter(file_name) as w:
      for s in random_sequences(self._N, 5, 5):
        example = tf.train.Example(features=tf.train.Features(
          feature={'sequence': int64_list_feature(s),
                   'length': int64_feature(len(s))}
        ))
        w.write(example.SerializeToString())
    return file_name

  def test_basic(self):
    tf_path = self.write_examples()
    feature_map = {
      'sequence': tf.VarLenFeature(tf.int64),
      'length': tf.FixedLenFeature([1], tf.int64)
    }
    dataset = Pipeline(tf_path, feature_map, one_shot=False,
                       batch_size=self._batch_size)
    with self.test_session() as sess:
      sess.run(dataset.init_op)
      for i in range(int(self._N / self._batch_size) + 1):
        batch_v = sess.run(dataset.batch)
        self.assertEqual(batch_v['sequence'].shape[0], self._batch_size)

  def test_epochs(self):
    tf_path = self.write_examples()

    NUM_EPOCHS = 3
    N = self._N
    batch_size = self._batch_size
    batch_per_epoch = int(N / batch_size)
    total_batch = NUM_EPOCHS * batch_per_epoch

    feature_map = {
      'sequence': tf.VarLenFeature(tf.int64),
      'length': tf.FixedLenFeature([1], tf.int64)
    }
    dataset = Pipeline(tf_path, feature_map,
                       batch_size=self._batch_size,
                       num_epochs=NUM_EPOCHS, one_shot=True)
    with self.test_session() as sess:
      for i in range(int(total_batch)):
        batch_v = sess.run(dataset.batch)
        self.assertEqual(batch_v['sequence'].shape[0], self._batch_size)
      with self.assertRaises(tf.errors.OutOfRangeError):
        batch_v = sess.run(dataset.batch)


if __name__ == "__main__":
  tf.test.main()

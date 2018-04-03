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

def no_op_embedding(x, *args, **kwargs):
  """For use when an embedding function is required but the inputs
  do not need to be embedded, e.g., bag of words encoding.
  """

  rank = len(x.get_shape().as_list())
  if rank == 2:
    # convert token ids into single-element embeddings of the same value
    outputs = tf.expand_dims(x, 2)
    outputs = tf.cast(outputs, dtype=tf.float32)
  elif rank == 3:
    outputs = x
    outputs = tf.cast(outputs, dtype=tf.float32)
  else:
    raise ValueError("x has invalid rank. rank must be 2 or 3: rank=%d" %(rank))

  return outputs

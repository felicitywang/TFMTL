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

import tensorflow as tf


def no_op_encoding(inputs, lengths):
  """A no-op encoder, e.g., for use with bag-of-words features
  that have already been encoded.

  Inputs
  ------
    inputs: batch of size [batch_size, D]

  Outputs
  -------
    outputs: a Tensor of size [batch_size, D].
  """

  return inputs


def concat_extractor(inputs, lengths):
  if not isinstance(inputs, list):
    inputs = [inputs]
  return tf.concat(inputs, axis=1)  # concat along time axis

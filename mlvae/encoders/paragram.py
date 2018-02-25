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
from six.moves import xrange

from mlvae.reducers import *
from mlvae.layers import dense_layer


def paragram_phrase(inputs,
                    lengths,
                    reducer=reduce_avg_over_time,
                    apply_activation=False,
                    activation_fn=None):
  """Processes inputs using the paragram_phrase
  method of Wieting et al. (https://arxiv.org/abs/1511.08198)

  Inputs
  ------
    inputs: batch of size [batch_size, batch_Len, embed_size]
    lengths: batch of size [batch_size]
    reducer: pooling operation to apply to the word embeddings to get the sentence embedding
    apply_activation: whether to apply an activation function to the sentence embedding
    activation_fn: (non-)linearity to apply to the reduced sentence embedding (linear projection if activation_fn=None)

  Outputs
  -------
    If the input word vectors have dimension D, the output is a Tensor of size 
    [batch_size, D].
  """

  if len(lengths.get_shape()) == 1:
    lengths = tf.expand_dims(lengths, 1)

  s_embedding = reducer(inputs, lengths=lengths, time_axis=1)

  if apply_activation:
    embed_dim = tf.shape(inputs)[2]
    s_embedding = dense_layer(s_embedding, embed_dim, name="paragram_phrase", activation=activation_fn)

  return s_embedding

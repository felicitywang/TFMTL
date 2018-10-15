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


def embed_sequence(x, vocab_size, embed_dim):
  init = tf.contrib.layers.xavier_initializer(uniform=True)
  return tf.contrib.layers.embed_sequence(x,
                                          unique=True,  # save memory ?
                                          vocab_size=vocab_size,
                                          embed_dim=embed_dim,
                                          initializer=init)

  # weights = tf.fill(tf.shape(x), 1.0)
  # return embed_sequence_weighted(x, weights, vocab_size, embed_dim)


def embed_sequence_weighted(x, weights, vocab_size, embed_dim):
  init = tf.contrib.layers.xavier_initializer(uniform=True)

  assert x.get_shape().as_list() == weights.get_shape().as_list(), \
    "{} != {}".format(tf.shape(x).as_list(), tf.shape(weights).as_list())

  embeddings = tf.contrib.layers.embed_sequence(x,
                                                unique=True,  # save memory ?
                                                vocab_size=vocab_size,
                                                embed_dim=embed_dim,
                                                initializer=init)

  print(embeddings.get_shape())

  embeddings = tf.transpose(embeddings, perm=[0, 2, 1])
  print(embeddings.get_shape())

  weights = tf.expand_dims(weights, 1)
  print(weights.get_shape())

  embeddings = tf.multiply(embeddings, weights)
  print(embeddings.get_shape())

  embeddings = tf.transpose(embeddings, perm=[0, 2, 1])
  print(embeddings.get_shape())

  return embeddings


def main():
  vocab_size = 4
  embed_dim = 2

  # # test without batch
  # x = tf.constant([0, 1, 2])
  # # w = tf.constant([[0.2, 0.2, 0.6], [0.3, 0.3, 0.4]])
  # w = tf.constant([0.2, 0.2, 0.6])
  #
  # embedded = embed_sequence(x, vocab_size, embed_dim)
  # # print(embeddings)
  #
  # sess = tf.InteractiveSession()
  # sess.run(tf.global_variables_initializer())
  # initial_embeddings = sess.run(embedded)
  # print(initial_embeddings)
  # print(initial_embeddings.shape)
  #
  # initial_embeddings = sess.run(tf.transpose(initial_embeddings, perm=[1, 0]))
  # print(initial_embeddings)
  # print(initial_embeddings.shape)
  #
  # weighted_embeddings = tf.multiply(initial_embeddings, w)
  # weighted_embeddings = sess.run(tf.transpose(weighted_embeddings))
  # print(weighted_embeddings)
  # print(weighted_embeddings.shape)

  # test with batch
  x = tf.constant([[0, 1, 2], [0, 1, 2], [0, 1, 2]])

  w = tf.constant(
    [[0.333, 0.333, 0.333],
     [0.2, 0.2, 0.6],
     [0.3, 0.3, 0.4]])

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  # use embed_sequence and multiplies weights
  # embedded = embed_sequence(x, vocab_size, embed_dim)
  #
  # initial_embeddings = sess.run(embedded)
  # print(initial_embeddings)
  # print(initial_embeddings.shape)
  #
  # initial_embeddings = sess.run(tf.transpose(initial_embeddings,
  #                                            perm=[0, 2, 1]))
  # print(initial_embeddings)
  # print(initial_embeddings.shape)
  #
  # weighted_embeddings = tf.multiply(initial_embeddings, tf.expand_dims(w, 1))
  #
  # weighted_embeddings = sess.run(
  #   tf.transpose(weighted_embeddings, perm=[0, 2, 1]))
  # print(weighted_embeddings)
  # print(weighted_embeddings.shape)

  # use embed_sequence_weighted directly

  # another_weighted_embeddings = embed_sequence_weighted(x, w,
  #                                                       vocab_size,
  #                                                       embed_dim)
  # another_weighted_embeddings = sess.run(another_weighted_embeddings)
  # print(another_weighted_embeddings)
  # print(another_weighted_embeddings.shape)


if __name__ == '__main__':
  main()

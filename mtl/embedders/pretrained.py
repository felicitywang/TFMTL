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

import os

import glove
import numpy as np
import tensorflow as tf


# TODO other word embeddings
# TODO pretrained+train

def glove_only(x, vocab_size, embed_dim, glove_path, trainable):
  """Use pre-trained Glove word embedding only, embeddings not to be trained

  :param x: list of word ids
  :param vocab_size: size of the vocabulary given in the config file
  :param embed_dim: dimension of the embeddings given in the config file
  :param glove_path: path to the pre-trained word embedding file
  :param trainable: whether to train the pred-trained word embeddings
  :return: embed lookup layer
  """
  tf.logging.info('Loading glove embeddings from %s' % glove_path)
  word_embedding_matrix = glove.Glove.load_stanford(
    glove_path).word_vectors
  assert word_embedding_matrix.shape[
           0] == vocab_size, "Given vocab size (%d) and that of the " \
                             "pre-trained embedding (%d) don't match!" % (
                               vocab_size, word_embedding_matrix.shape[0])
  assert word_embedding_matrix.shape[
           1] == embed_dim, "Given embed dim (%d) and that of the " \
                            "pre-trained embedding (%d) don't match!" % (
                              embed_dim, word_embedding_matrix.shape[1])
  # glove file name - .txt
  word_embedding_name = os.path.basename(glove_path)[:-4]
  tf.logging.info('Generating embedding lookup layer from %s' % glove_path)
  word_embedding_lookup = tf.get_variable(
    name=word_embedding_name,
    initializer=tf.constant_initializer(np.float32(word_embedding_matrix)),
    dtype=tf.float32,
    shape=[vocab_size, embed_dim],
    trainable=trainable)

  return tf.nn.embedding_lookup(word_embedding_lookup,
                                x)


def glove_and_train(x, vocab_size, embed_dim, glove_path, trainable):
  """Use pre-trained Glove word embedding only, embeddings not to be trained

  :param x: list of word ids
  :param vocab_size: size of the vocabulary given in the config file
  :param embed_dim: dimension of the embeddings given in the config file
  :param glove_path: path to the pre-trained word embedding file
  :param trainable: whether to train the pred-trained word embeddings
  :return: embed lookup layer
  """
  tf.logging.info('Loading glove embeddings from %s' % glove_path)
  word_embedding_matrix = glove.Glove.load_stanford(
    glove_path).word_vectors
  assert word_embedding_matrix.shape[
           0] <= vocab_size, "Given vocab size (%d) is less than that of " \
                             "the " \
                             "pre-trained embedding (%d)!" % (
                               vocab_size, word_embedding_matrix.shape[0])
  assert word_embedding_matrix.shape[
           1] == embed_dim, "Given embed dim (%d) and that of the " \
                            "pre-trained embedding (%d) don't match!" % (
                              embed_dim, word_embedding_matrix.shape[1])
  # glove file name - .txt
  # word_embedding_name = os.path.basename(glove_path)[:-4]
  tf.logging.info('Generating embedding lookup layer from %s and the words '
                  'from the training set' %
                  glove_path)

  pretrained_embedding = tf.get_variable(
    name='embedding_pretrained',
    initializer=tf.constant_initializer(np.float32(word_embedding_matrix)),
    dtype=tf.float32,
    shape=[word_embedding_matrix.shape[0], embed_dim],
    trainable=trainable)

  # randomly initialize word embeddings for words that appear in the
  # training set but not in pre-trained word embeddings
  extra_vocab_num = vocab_size - word_embedding_matrix.shape[0]
  print('There are %d words in the training set(s) that are not found in the '
        'pre-trained word embedding dictionary. '
        'Randomly initializing word embeddings for them...' % extra_vocab_num)

  extra_embedding = tf.get_variable(
    name='embedding_training',
    initializer=tf.random_uniform(shape=[extra_vocab_num, embed_dim],
                                  dtype=tf.float32),
    dtype=tf.float32,
    trainable=True
  )

  word_embedding = tf.concat([pretrained_embedding,
                              extra_embedding],
                             axis=0,
                             name='embedding_combined')

  assert word_embedding.shape.as_list() == [vocab_size, embed_dim]

  return tf.nn.embedding_lookup(word_embedding,
                                x)

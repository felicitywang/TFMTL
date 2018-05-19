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

import codecs
import json

import numpy as np
import tensorflow as tf
# TODO other word embeddings
from tqdm import tqdm


def expand_glove(x, vocab_size, embed_dim, glove_path, trainable):
  """Expand training vocab with Glove

  :param x: list of word ids
  :param vocab_size: size of the vocabulary given in the config file
  :param embed_dim: dimension of the embeddings given in the config file
  :param glove_path: path to the pre-trained word embedding file
  :param trainable: whether to train the pred-trained word embeddings
  :return: embed lookup layer
  """
  import glove
  tf.logging.info('Loading glove embeddings from %s' % glove_path)
  pretrained_matrix = glove.Glove.load_stanford(
    glove_path).word_vectors
  assert pretrained_matrix.shape[
           0] <= vocab_size, "Given vocab size (%d) is less than that of " \
                             "the " \
                             "pre-trained embedding (%d)!" % (
                               vocab_size, pretrained_matrix.shape[0])
  assert pretrained_matrix.shape[
           1] == embed_dim, "Given embed dim (%d) and that of the " \
                            "pre-trained embedding (%d) don't match!" % (
                              embed_dim, pretrained_matrix.shape[1])
  # glove file name - .txt
  # word_embedding_name = os.path.basename(glove_path)[:-4]
  tf.logging.info('Generating embedding lookup layer from %s and the words '
                  'from the training set' %
                  glove_path)

  loaded_embedding = tf.get_variable(
    name='embedding_pretrained',
    initializer=tf.constant_initializer(np.float32(pretrained_matrix)),
    dtype=tf.float32,
    shape=[pretrained_matrix.shape[0], embed_dim],
    trainable=trainable)

  # randomly initialize word embeddings for words that appear in the
  # training set but not in pre-trained word embeddings
  extra_vocab_num = vocab_size - pretrained_matrix.shape[0]
  print('There are %d words in the training set(s) that are not found in the '
        'pre-trained word embedding dictionary. '
        'Randomly initializing word embeddings for them...' % extra_vocab_num)

  random_embedding = tf.get_variable(
    name='embedding_training',
    initializer=tf.random_uniform(shape=[extra_vocab_num, embed_dim],
                                  dtype=tf.float32),
    dtype=tf.float32,
    trainable=True
  )

  word_embedding = tf.concat([random_embedding, loaded_embedding],
                             axis=0,
                             name='embedding_combined')

  assert word_embedding.shape.as_list() == [vocab_size, embed_dim]

  return tf.nn.embedding_lookup(word_embedding,
                                x)


def init_glove(x, vocab_size, embed_dim, glove_path, reverse_vocab_path,
               random_size_path, trainable):
  """Initialize training vocab with Glove's pre-trained word embeddings,
  always trainable

  :param x: list of word ids
  :param vocab_size: size of the vocabulary given in the config file
  :param embed_dim: dimension of the embeddings given in the config file
  :param glove_path: path to the pre-trained word embedding file
  :param reverse_vocab_path: path to the vocab file(id to word mapping of
  all the dictionary(extra train + Glove))
  :param trainable: whether to fine-tune the part of word embeddings
  initialized from Glove
  :param: random_size_path: path to the size of word embeddings to be
  randomly initialized(those not in Glove)
  :return: embed lookup layer
  """
  import glove
  with codecs.open(random_size_path) as file:
    random_size = int(file.readline().split()[0])

  tf.logging.info('Randomly initializing word embeddings for %s words not '
                  'in Glove...' % random_size)

  random_embedding = tf.get_variable(
    name='embedding_training',
    initializer=tf.random_uniform(shape=[random_size, embed_dim],
                                  dtype=tf.float32),
    dtype=tf.float32,
    trainable=True
  )

  # load training vocab
  with codecs.open(reverse_vocab_path) as file:
    reverse_vocab = json.load(file)

  tf.logging.info('Loading glove embeddings from %s' % glove_path)
  pretrained_embedding = glove.Glove.load_stanford(glove_path)
  pretrained_vocab = pretrained_embedding.dictionary
  pretrained_matrix = pretrained_embedding.word_vectors

  assert pretrained_matrix.shape[
           1] == embed_dim, "Given embed dim (%d) and that of the " \
                            "pre-trained embedding (%d) don't match!" % (
                              embed_dim, pretrained_matrix.shape[1])

  # glove file name - .txt
  # word_embedding_name = os.path.basename(glove_path)[:-4]
  loaded_matrix = np.zeros([vocab_size - random_size, embed_dim])

  for i in tqdm(range(random_size, len(reverse_vocab))):
    v = reverse_vocab[str(i)]
    loaded_matrix[i - random_size] = pretrained_matrix[
      pretrained_vocab.get(v)]

  loaded_embedding = tf.get_variable(
    name='embedding_pretrained',
    initializer=tf.constant_initializer(np.float32(loaded_matrix)),
    dtype=tf.float32,
    shape=[vocab_size - random_size, embed_dim],
    trainable=trainable)

  tf.logging.info('Generating embedding lookup layer from %s and the words '
                  'from the training set' %
                  glove_path)
  word_embedding = tf.concat([random_embedding, loaded_embedding],
                             axis=0,
                             name='embedding_combined')

  assert word_embedding.shape.as_list() == [vocab_size, embed_dim]

  return tf.nn.embedding_lookup(word_embedding,
                                x)

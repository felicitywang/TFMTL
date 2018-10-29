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

from mtl.embedders.embed_sequence import get_weighted_embeddings
from mtl.util.load_embeds import (load_pretrained_matrix,
                                  load_pretrianed_vocab_dict)


# TODO add weights

def expand_pretrained(word_ids,
                      vocab_size,
                      embed_dim,
                      pretrained_path,
                      trainable,
                      **kwargs):
  """Expand training vocab with pretrained

  :param word_ids: list of word ids
  :param vocab_size: size of the vocabulary given in the config file
  :param embed_dim: dimension of the embeddings given in the config file
  :param pretrained_path: path to the pre-trained word embedding file
  :param trainable: whether to train the pred-trained word embeddings
  :return: embed lookup layer
  """

  if kwargs['is_training']:

    tf.logging.info('Loading pretrained embeddings from %s' %
                    pretrained_path)
    pretrained_matrix = load_pretrained_matrix(pretrained_path)
    assert pretrained_matrix.shape[
             0] <= vocab_size, "Given vocab size (%d) is less than that of " \
                               "the " \
                               "pre-trained embedding (%d)!" % (
                                 vocab_size, pretrained_matrix.shape[0])
    assert pretrained_matrix.shape[
             1] == embed_dim, "Given embed dim (%d) and that of the " \
                              "pre-trained embedding (%d) don't match!" % (
                                embed_dim, pretrained_matrix.shape[1])
    # pretrained file name - .txt
    # word_embedding_name = os.path.basename(pretrained_path)[:-4]
    tf.logging.info('Generating embedding lookup layer from %s and the words '
                    'from the training set' %
                    pretrained_path)

    loaded_embedding = tf.get_variable(
      name='embedding_pretrained',
      initializer=tf.constant_initializer(np.float32(pretrained_matrix)),
      dtype=tf.float32,
      shape=[pretrained_matrix.shape[0], embed_dim],
      trainable=trainable)

    # randomly initialize word embeddings for words that appear in the
    # training set but not in pre-trained word embeddings
    extra_vocab_num = vocab_size - pretrained_matrix.shape[0]
    print(
      'There are %d words in the training set(s) that are not found in the '
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

    embeddings = tf.contrib.layers.embedding_lookup_unique(word_embedding,
                                                           word_ids)

    if 'weights' in kwargs:
      embeddings = get_weighted_embeddings(embeddings,
                                           weights=kwargs['weights'])

    return embeddings

  else:
    # not initializing again but only define placeholders with same names
    tf.logging.info('Loading pretrained embeddings from %s' %
                    pretrained_path)
    pretrained_matrix = load_pretrained_matrix(pretrained_path)
    assert pretrained_matrix.shape[
             0] <= vocab_size, "Given vocab size (%d) is less than that of " \
                               "the " \
                               "pre-trained embedding (%d)!" % (
                                 vocab_size, pretrained_matrix.shape[0])
    assert pretrained_matrix.shape[
             1] == embed_dim, "Given embed dim (%d) and that of the " \
                              "pre-trained embedding (%d) don't match!" % (
                                embed_dim, pretrained_matrix.shape[1])

    # pretrained_matrix = pretrained_matrix.fill(1)

    # pretrained file name - .txt
    # word_embedding_name = os.path.basename(pretrained_path)[:-4]
    tf.logging.info('Initializing embedding lookup layer from %s and the '
                    'words from the training set' %
                    pretrained_path)
    loaded_embedding = tf.get_variable(
      name='embedding_pretrained',
      # initializer=tf.constant_initializer(np.float32(pretrained_matrix)),
      initializer=tf.zeros(shape=[pretrained_matrix.shape[0], embed_dim],
                           dtype=tf.float32),
      dtype=tf.float32,
      # shape=[pretrained_matrix.shape[0], embed_dim],
      trainable=trainable
    )

    # randomly initialize word embeddings for words that appear in the
    # training set but not in pre-trained word embeddings
    extra_vocab_num = vocab_size - pretrained_matrix.shape[0]
    print(
      'There are %d words in the training set(s) that are not found in the '
      'pre-trained word embedding dictionary. '
      'Randomly initializing word embeddings for them...' % extra_vocab_num)

    random_embedding = tf.get_variable(
      name='embedding_training',
      # initializer=tf.random_uniform(shape=[extra_vocab_num, embed_dim],
      #                               dtype=tf.float32),
      initializer=tf.zeros(shape=[extra_vocab_num, embed_dim],
                           dtype=tf.float32),
      dtype=tf.float32,
      trainable=True
    )

    word_embedding = tf.concat([random_embedding, loaded_embedding],
                               axis=0,
                               name='embedding_combined')

    assert word_embedding.shape.as_list() == [vocab_size, embed_dim]

    embeddings = tf.contrib.layers.embedding_lookup_unique(word_embedding,
                                                           word_ids)

    if 'weights' in kwargs:
      embeddings = get_weighted_embeddings(embeddings,
                                           weights=kwargs['weights'])

    return embeddings


def init_pretrained(word_ids,
                    vocab_size,
                    embed_dim,
                    pretrained_path,
                    reverse_vocab_path,
                    random_size,
                    trainable,
                    **kwargs):
  """Initialize training vocab with pretrained's pre-trained word embeddings,
  always trainable

  :param word_ids: list of word ids
  :param vocab_size: size of the vocabulary given in the config file
  :param embed_dim: dimension of the embeddings given in the config file
  :param pretrained_path: path to the pre-trained word embedding file
  :param reverse_vocab_path: path to the vocab file(id to word mapping of
  all the dictionary(extra train + pretrained))
  :param trainable: whether to fine-tune the part of word embeddings
  initialized from pretrained
  :param random_size: size of word embeddings to be
  randomly initialized(those not in pretrained)
  :return: embed lookup layer
  """

  tf.logging.info('Randomly initializing word embeddings for %s words not '
                  'in pretrained...' % random_size)

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

  tf.logging.info('Loading pretrained embeddings from %s' % pretrained_path)
  pretrained_vocab = load_pretrianed_vocab_dict(pretrained_path)
  pretrained_matrix = load_pretrained_matrix(pretrained_path)

  assert pretrained_matrix.shape[
           1] == embed_dim, "Given embed dim (%d) and that of the " \
                            "pre-trained embedding (%d) don't match!" % (
                              embed_dim, pretrained_matrix.shape[1])

  # pretrained file name - .txt
  # word_embedding_name = os.path.basename(pretrained_path)[:-4]
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
                  pretrained_path)
  word_embedding = tf.concat([random_embedding, loaded_embedding],
                             axis=0,
                             name='embedding_combined')

  assert word_embedding.shape.as_list() == [vocab_size, embed_dim]

  embeddings = tf.contrib.layers.embedding_lookup_unique(word_embedding,
                                                         word_ids)

  if 'weights' in kwargs:
    embeddings = get_weighted_embeddings(
      embeddings, weights=kwargs['weights'])

  return embeddings

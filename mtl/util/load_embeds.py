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
# =============================================================================

# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from tqdm import tqdm


def combine_vocab(glove_path, train_vocab_list):
  """Expand the training vocab with all the words in Glove. Modified from

  https://gitlab.hltcoe.jhu.edu/research/tf-ner/blob/master/ner/embeds.py

  :param glove_path: path to the Glove file
  :param train_vocab_list: list, all the word types in the training data
  :return: glove vocab + training vocab
  """
  import glove
  print('Loading embeddings from {}...\n'.format(glove_path), end='')
  glove_vocab_dict = glove.Glove.load_stanford(glove_path).dictionary  # used
  #  to create the final vocab and keep order
  glove_vocab_set = set(glove_vocab_dict)  # used to look for train vocab

  print('{} original vocabulary from training.'.format(len(train_vocab_list)))
  print('{} original vocabulary from glove.'.format(len(glove_vocab_dict)))

  train_vocab_set = set(train_vocab_list)

  extra_vocab_set = set()
  for v in tqdm(train_vocab_set):
    if v not in glove_vocab_set:
      extra_vocab_set.add(v)

  if '<UNK>' in extra_vocab_set:
    extra_vocab_list = [v for v in tqdm(train_vocab_list) if
                        v in extra_vocab_set]
  else:
    extra_vocab_list = ['<UNK>'] + [v for v in tqdm(train_vocab_list) if
                                    v in extra_vocab_set]

  assert extra_vocab_list[0] == '<UNK>'

  print('{} words in training vocab not in pre-trained word embedding '
        'dictionary.'.format(len(extra_vocab_set)))

  combined_vocab = {w: i for i, w in
                    enumerate(extra_vocab_list + list(glove_vocab_dict))}

  return combined_vocab, extra_vocab_list


def reorder_vocab(glove_path, training_vocab_list):
  """Reorder training vocab to [not in Glove, in Glove]

  :param glove_path: path to the Glove file
  :param train_vocab_list: list, all the word types in the training data
  :return: reordered vocab
  """
  import glove
  print('Loading embeddings from {}...\n'.format(glove_path))
  glove_vocab_set = set(glove.Glove.load_stanford(glove_path).dictionary)

  not_in_glove = []
  in_glove = []

  for v in training_vocab_list:
    if v in glove_vocab_set:
      in_glove.append(v)
    else:
      not_in_glove.append(v)

  # len(not_in_glove) is the size of word embeddings to be randomly initialized
  return len(not_in_glove), {w: i for i, w in
                             enumerate(not_in_glove + in_glove)}

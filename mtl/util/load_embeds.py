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

import glove
from tqdm import tqdm


def load_glove(glove_path, train_vocab_set):
  """Combine Glove vocabulary and training-data vocabulary, modified from

  https://gitlab.hltcoe.jhu.edu/research/tf-ner/blob/master/ner/embeds.py

  :param glove_path: path to the Glove file
  :param train_vocab_set: set, all the word types in the training data
  :return: glove vocab + training vocab
  """
  glove_vocab_dict = glove.Glove.load_stanford(glove_path).dictionary  # used
  #  to create the final vocab and keep order
  glove_vocab_set = set(glove_vocab_dict)  # used to look for train vocab

  print('Loading embeddings from {}...\n'.format(glove_path), end='')
  print('{} original vocabulary from training.'.format(len(train_vocab_set)))
  print('{} original vocabulary from glove.'.format(len(glove_vocab_dict)))

  extra_vocab = set()
  for v in tqdm(train_vocab_set):
    if v not in glove_vocab_set:
      extra_vocab.add(v)
  extra_vocab = list(extra_vocab)

  print('{} words in training vocab not in pre-trained word embedding '
        'dictionary.'.format(len(extra_vocab)))

  combined_vocab = {w: i for i, w in
                    enumerate(list(glove_vocab_dict) + extra_vocab)}

  return combined_vocab, extra_vocab

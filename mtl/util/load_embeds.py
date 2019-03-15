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

"""Code to load pretrained word embeddings and merge the vocabulary with

that of the training dataset(s)"""

from __future__ import unicode_literals

import array
import io
from zipfile import ZipFile

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm


def combine_vocab(pretrained_path, train_vocab_list):
    """Expand the training vocab with the pretrained embedding vocabulary.

    Modified from
    https://gitlab.hltcoe.jhu.edu/research/tf-ner/blob/master/ner/embeds.py

    :param pretrained_path: path to the pretrained word embedding file
    :param train_vocab_list: list, all the word types in the training data
    :return: pretrained vocab + training vocab
    """
    pretrained_vocab_dict = load_pretrianed_vocab_dict(
        pretrained_path)  # used to create the
    # final vocab and keep order
    pretrained_vocab_dict = set(
        pretrained_vocab_dict)  # used to look for train vocab

    print(
        '{} original vocabulary from training data.'.format(
            len(train_vocab_list)))
    print('{} original vocabulary from pretrained word embedding.'.format(len(
        pretrained_vocab_dict)))

    train_vocab_set = set(train_vocab_list)

    extra_vocab_set = set()
    for v in tqdm(train_vocab_set):
        if v not in pretrained_vocab_dict:
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
                      enumerate(extra_vocab_list + list(pretrained_vocab_dict))}

    return combined_vocab, extra_vocab_list


def reorder_vocab(pretrained_path, training_vocab_list):
    """Reorder training vocab to [not in pretrained, in pretrained]

    :param pretrained_path: path to the pretrained word embedding file
    :param training_vocab_list: list, all the word types in the training data
    :return: reordered vocab
    """
    pretrained_vocab_dict = load_pretrianed_vocab_dict(
        pretrained_path)  # used to create the
    # final vocab and keep order
    pretrained_vocab_set = set(
        pretrained_vocab_dict)  # used to look for train vocab

    not_in_pretrained = []
    in_pretrained = []

    for v in training_vocab_list:
        if v in pretrained_vocab_set:
            in_pretrained.append(v)
        else:
            not_in_pretrained.append(v)

    # len(not_in_pretrained) is the size of word embeddings to be randomly
    # initialized
    return len(not_in_pretrained), {w: i for i, w in
                                    enumerate(
                                        not_in_pretrained + in_pretrained)}


def load_pretrained_matrix(filepath):
    """Load the pretrained word embedding matrix.

    :param filepath: full file path of the pretrained embedding file
    :return: np array, pretrained word vectors.
    """
    print('Loading embedding matrix from {}...'.format(filepath))

    num = 0
    if filepath.endswith('.bin.gz'):  # word2vec
        word_vectors = []
        model = KeyedVectors.load_word2vec_format(filepath, binary=True)
        vocab_list = {v: i for i, v in enumerate(list(model.vocab.keys()))}
        for v in vocab_list:
            word_vectors.append(model.get_vector(v))
        word_vectors = np.array(word_vectors).reshape(len(word_vectors),
                                                      len(word_vectors[0]))
    elif filepath.endswith('.txt'):  # glove
        word_vectors = array.array('d')
        with io.open(filepath, 'r', encoding='utf-8') as file:
            for _, line in tqdm(enumerate(file)):
                entries = line.split(' ')[1:]
                word_vectors.extend(float(x) for x in entries)
                num += 1
        dim = len(entries)
        word_vectors = (np.array(word_vectors)
                        .reshape(num, dim))
    elif filepath.endswith('.zip'):  # fasttext
        word_vectors = array.array('d')
        with ZipFile(filepath, 'r') as myzip:
            with myzip.open(filepath[filepath.rfind('/') + 1:filepath.find(
                '.zip')]) as file:
                num, dim = map(int, file.readline().split())
                for _, line in tqdm(enumerate(file.readlines())):
                    entries = line.decode('utf-8').split(' ')[1:]
                    word_vectors.extend(float(x) for x in entries)
                word_vectors = (np.array(word_vectors)
                                .reshape(num, dim))
    else:
        raise ValueError('No such embedding file as {}!'.format(filepath))

    return word_vectors


def load_pretrianed_vocab_dict(filepath):
    """Load the pretrained word embedding vocab dictionary.

    :param filepath: full file path of the pretrained embedding file
    :return: dict, pretrained word id mapping.
    """
    print('Loading pretrained embedding dictionary from {}...'.format(filepath))

    if 'bin' in filepath:
        vocab_dict = {v: i for i, v in enumerate(list(
            KeyedVectors.load_word2vec_format(
                filepath,
                binary='bin' in filepath).vocab.keys()))}
    elif filepath.endswith('.txt'):
        vocab_dict = {}
        with io.open(filepath, 'r', encoding='utf-8') as file:
            for i, line in tqdm(enumerate(file)):
                delimiter = '\t' if '\t' in line else ' '
                vocab_dict[line.split(delimiter)[0]] = i
    elif filepath.endswith('.zip'):  # fasttext
        vocab_dict = {}
        with ZipFile(filepath, 'r') as myzip:
            with myzip.open(filepath[filepath.rfind('/') + 1:filepath.find(
                '.zip')]) as file:
                num, dim = map(int, file.readline().split())
                for i, line in tqdm(enumerate(file.readlines())):
                    vocab_dict[line.decode('utf-8').split(' ')[0]] = i
                assert len(vocab_dict) == num
    else:
        raise ValueError('No such embedding file as {}!'.format(filepath))
    return vocab_dict


# load fasttext
# https://fasttext.cc/docs/en/english-vectors.html
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

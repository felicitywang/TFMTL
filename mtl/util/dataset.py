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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import functools
import gzip
import itertools
import json
import operator
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from six.moves import xrange
from tqdm import tqdm

from mtl.util.categorical_vocabulary import CategoricalVocabulary
from mtl.util.constants import OLD_LINEBREAKS, LINEBREAK, EOS, BOS, OOV
from mtl.util.constants import RANDOM_SEED
from mtl.util.constants import TRAIN_RATIO, VALID_RATIO
from mtl.util.constants import VOCAB_NAMES
from mtl.util.data_prep import (tweet_tokenizer,
                                tweet_tokenizer_keep_handles,
                                ruder_tokenizer,
                                split_tokenizer, lower_tokenizer, preproc,
                                remove_stopwords, porter_stemmer,
                                snowball_stemmer, wordnet_stemmer)
from mtl.util.load_embeds import combine_vocab, reorder_vocab, \
  load_pretrianed_vocab_dict
from mtl.util.text import VocabularyProcessor, tokenizer_simple
from mtl.util.util import bag_of_words, tfidf, make_dir

flags = tf.flags
logging = tf.logging

FLAGS = flags.FLAGS


class Dataset:
  def __init__(self,
               json_dir,
               vocab_given,
               generate_basic_vocab,
               generate_tf_record,
               tfrecord_dir,
               vocab_dir,
               vocab_name='vocab_freq.json',
               predict_json_path='predict.json.gz',
               predict_tf_path='predict.tf',
               **kwargs):

    """TODO Args:

    :param json_dir: where data.json.gz and index.json.gz are
            located, and vocabulary/TFRecords built from the single
            datasets are to be saved
    :param vocab_given: True if to use the given vocabulary
    :param tfrecord_dir: directory to save generated TFRecord files
    :param vocab_dir: directory to load the given (e.g. merged) vocabulary
            frequency dict json file or to save the self-generated vocabulary
    :param vocab_name: 'vocab_freq.json' or 'vocab_v2i.json'
    :param max_document_length: maximum document length for the mapped
            word ids, computed as the maximum document length of all the
            training data if -1, used when padding is True
    :param max_vocab_size: maximum size of the vocabulary allowed
    :param min_frequency: minimum frequency to build the vocabulary,
            words that appear lower than or equal to this number would be
            discarded
    :param max_frequency: maximum frequency to build the vocabulary,
            words that appear more than or equal to this number would be
            discarded
    :param text_field_names: string of a list of text field names joined
            with spaces, read from json_dir if None
    :param label_field_name: label field name(only 1), read from json_dir if None
    :param label_type: type of label, 'int' or 'float'
    :param valid_ratio: how many data out of all data to use as valid
            data if not splits are given, or how many data out of train data to
            use as valid if train/test splits are given
    :param train_ratio: how many data to use as train data if train
            split not given
    :param random_seed: random seed used in random spliting, making
            sure the same random split is used when given the same random seed
    :param subsample_ratio: randomly takes part of the datasets when it's
            too large
    :param generate_basic_vocab: True if the basic vocabulary(which
            shall be used to merge the public vocabulary) needs to be generated
    :param generate_tf_record: True if TFRecord files need generating
    :param padding: True if token(word id) list needs padding to
            max_document_length
    :param write_bow: True if to write bag of words as a feature in the TFRecord
    :param write_tfidf: True if to write tf-idf as a feature in the TFRecord
    :param predict_mode: True if to only write unlabeled text to predict
    :param predict_json_path: File path of the gzipped json file of the text to
        predict
    :param predict_tf_path: File path of the TFRecord file of the text to predict
    :param expand_vocab: whether to expand the training vocab with pre-trained
        word embeddings' vocab
    :param preproc: whether to remove urls, trailing/leading whitespaces and
        replace linebreaks
    :param vocab_all: whether to use all three splits when building vocabulary
    """

    self._json_dir = json_dir
    self._tfrecord_dir = tfrecord_dir
    self._load_vocab_name = vocab_name
    self._predict_tf_path = predict_tf_path

    self._args = {
      'predict_mode': False,
      'text_field_names': ['text'],
      'label_field_name': ['label'],
      'label_type': 'int',
      'max_document_length': -1,
      'min_frequency': 0,
      'max_frequency': -1,
      'max_vocab_size': -1,
      'tokenizer_': 'tweet_tokenizer',
      'stemmer': None,
      'pretrained_only': False,
      'expand_vocab': False,
      'preproc': True,
      'stopwords': 'nltk',
      'vocab_all': False,
      'padding': False,
      'write_bow': False,
      'write_tfidf': False,
      'train_ratio': TRAIN_RATIO,
      'valid_ratio': VALID_RATIO,
      'random_seed': RANDOM_SEED,
      'subsample_ratio': 1
    }
    for k, v in kwargs.items():
      # print(k)
      self._args[k] = v

    if self._args['max_document_length'] == -1:
      self._args['max_document_length'] = float('inf')

    # print(self._args)

    # used to generate word id mapping from word frequency dictionary and
    # arguments(min_frequency, max_frequency, max_document_length)
    if not generate_basic_vocab and not generate_tf_record \
      and vocab_given and vocab_name == 'vocab_freq.json':
      print(
        "Generating word id mapping using given word frequency dictionary...")
      self._vocab_dir = vocab_dir
      self._categorical_vocab = self.load_make_vocab()
      self._args['vocab_size'] = len(self._categorical_vocab.mapping)
      print("Vocabulary size =", self._args['vocab_size'])
      return

    if self._args['predict_mode']:
      assert predict_json_path is not None
      self._data_file_name = predict_json_path
    else:
      assert json_dir is not None
      self._data_file_name = os.path.join(json_dir, 'data.json.gz')

    self._tokenizer = None
    self.get_tokenizer()

    self._stemmer = None
    self.get_stemmer()

    self._data = None
    self.read_data()

    self._label_list = None
    self._label_set = None
    self._num_classes = None
    self.get_label()

    self._num_examples = 0
    self._sequences = dict()
    self._sequence_lengths = dict()
    self._ids = []
    self._weights = dict()
    self.get_text()

    self.get_index()
    self.get_max_doc_len()  # TODO remove?

    self._vocab_dict = None
    self._categorical_vocab = None
    self._vocab_freq_dict = None
    self._vocab_v2i_dict = None
    self._vocab_i2v_dict = None
    self._vocab_dir = None
    self._save_vocab_dir = None
    self._vocab_processor = None

    # generate and save the vocabulary which contains all the words
    if generate_basic_vocab:
      print("Generating the basic vocabulary.")
      self._vocab_dir = json_dir
      self.build_save_basic_vocab()

    if generate_tf_record is False:
      print("No need to generate TFRecords. Done.")
      return

    if vocab_given is False:
      print("No vocabulary given. Generate a new one.")
      self._categorical_vocab = self.build_vocab()
      self._vocab_dir = tfrecord_dir
      self._save_vocab_dir = self._vocab_dir
      self.save_vocab()

    else:
      print("Public vocabulary given. Use that to build vocabulary "
            "processor.")
      assert vocab_name in VOCAB_NAMES, vocab_name
      self._vocab_dir = vocab_dir
      self._tfrecord_dir = tfrecord_dir  # used to save the combined
      # vocabulary when loading pretrained word embeddings
      self._categorical_vocab = self.load_vocab()

    # save mapping/reverse mapping to the disk
    # freq:            vocab_dir/vocab_freq.json
    # mapping:         vocab_dir/vocab_v2i.json
    # reverse mapping: vocab_dir/vocab_i2v.json(sorted according to freq)

    self._args['vocab_size'] = len(self._categorical_vocab.mapping)
    print("used vocab size =", self._args['vocab_size'])

    # generate tfidf list if write_tfidf is True
    if self._args['write_tfidf']:
      # TODO: determine if tfidf should be calculated based on each
      # sequence kind or on all sequences regardless of kind
      raise NotImplementedError("tfidf (%s) not currently supported" % tfidf)

    self.write_tfrecord()

    if self._args['predict_mode']:
      self._tfrecord_dir = os.path.dirname(predict_tf_path)
    self.write_args()

    self.save_vocab()

  def get_max_doc_len(self):
    # TODO remove ?
    # only compute from training data
    if self._args['max_document_length'] == -1:
      if self._args['padding']:
        print('Maximum document length not given, computing from training '
              'data..')
        for text_field_name in tqdm(self._args['text_field_names']):
          tmp_max = max(len(self._sequences[text_field_name][i])
                        for i in self._train_index)
        self._args['max_document_length'] = max(tmp_max,
                                                self._args[
                                                  'max_document_length'])
        print("Max document length computed =",
              self._args['max_document_length'])
      else:
        self._args['max_document_length'] = float('inf')
        print('Maximum document length given: ',
              self._args['max_document_length'])
    else:
      print('Maximum document length given:',
            self._args['max_document_length'])

  def get_index(self):
    if self._args['predict_mode']:
      self._predict_index = np.asarray(range(self._num_examples))
    else:
      assert self._ids == []
      print("Generating train/valid/test splits...")
      index_path = os.path.join(self._json_dir, "index.json.gz")
      (self._train_index, self._valid_index, self._test_index,
       self._unlabeled_index) = self.split(index_path,
                                           self._args['train_ratio'],
                                           self._args['valid_ratio'],
                                           self._args['random_seed'],
                                           self._args['subsample_ratio']
                                           )

  def write_tfrecord(self):

    # write TFRecords for predict data
    if self._args['predict_mode']:
      self._args['predict_path'] = self._predict_tf_path
      print("Writing TFRecord file for the predicting file...")
      self.write_examples(self._args['predict_path'], self._predict_index,
                          labeled=False)
      return

    # write TFRecords for train/valid/test data

    # write labeled data to TFRecord files
    make_dir(self._tfrecord_dir)
    self._args['train_path'] = os.path.join(self._tfrecord_dir, 'train.tf')
    self._args['valid_path'] = os.path.join(self._tfrecord_dir, 'valid.tf')
    self._args['test_path'] = os.path.join(self._tfrecord_dir, 'test.tf')

    print("Writing TFRecord file for the training data...")
    self.write_examples(
      self._args['train_path'], self._train_index, labeled=True)
    print("Writing TFRecord file for the validation data...")
    self.write_examples(
      self._args['valid_path'], self._valid_index, labeled=True)
    print("Writing TFRecord file for the test data...")
    self.write_examples(
      self._args['test_path'], self._test_index, labeled=True)

    # write unlabeled data to TFRecord files if there're any

    if len(self._unlabeled_index) == 0:
      print("Unlabeled data not found.")
      self._args['unlabeled_path'] = None
      self._args['has_unlabeled'] = False
    else:
      print("Unlabeled data found.")
      self._args['_has_unlabeled'] = True
      print("Writing TFRecord files for the unlabeled data...")
      self._args['unlabeled_path'] = os.path.join(self._tfrecord_dir,
                                                  'unlabeled.tf')
      self.write_examples(
        self._args['unlabeled_path'], self._unlabeled_index, labeled=False)

  def get_label(self):
    print('Generating label list...')
    if self._args['label_type'] == 'int':
      transfer = int
    elif self._args['label_type'] == 'float':
      transfer = float
    else:
      raise TypeError('Label type other than "int" and "float" is not '
                      'implemetned!')
    self._label_list = [transfer(item[self._args['label_field_name']])
                        if self._args['label_field_name'] in item else None for
                        item in tqdm(self._data)]
    # TODO: is it possible that the label list read doesn't cover all the
    # labels?
    self._label_set = set(self._label_list)
    self._args['labels'] = list(self._label_set)
    self._num_classes = len(set(self._label_list))

  def read_data(self):
    print('Loading data from', self._data_file_name)
    with gzip.open(self._data_file_name, mode='rt') as file:
      self._data = json.load(file, encoding='utf-8')

  def get_text(self):
    # tokenize and reconstruct as string(which vocabulary processor
    # takes as input)

    # Iterate through data in the following order:
    #   -data example
    #   -type of sequence (e.g., "seq1", "seq2")
    # This makes it easy to keep the various sequences from a given example
    # "in line" with each other, meaning that, e.g.:
    #   self._sequences["seq1"][37]
    #   self._sequences["seq2"][37]
    # are from the same example in the data
    #
    # When the training, validation, and test data are determined by choosing
    # the indices of the examples for each group, respectively, we can
    # make sure that examples stay together by iterating over
    # the text_field_names and selecting
    #   self._sequences[text_field_name][index_of_interest]
    # for each text_field_name because the sequences for an example are
    # indexed by the same number.
    for text_field_name in self._args['text_field_names']:
      self._sequences[text_field_name] = list()
      self._sequence_lengths[text_field_name] = list()

    print("Generating text lists...")
    min_seq_len = sys.maxsize
    for index, item in tqdm(enumerate(self._data)):
      self._num_examples += 1

      if self._args['predict_mode']:
        if 'id' in item:
          self._ids.append(item['id'])

      for text_field_name in self._args['text_field_names']:
        text = item[text_field_name]

        if self._args['preproc']:

          # remove leading and trailing whitespaces(to get rid of redundant
          # linebreaks)
          text = text.strip()
          text = preproc(text)

          # replace line breaks
          for old_linebreak in OLD_LINEBREAKS:
            text = text.replace(old_linebreak, LINEBREAK)

        text = self._tokenizer(text)
        if self._stemmer:
          text = self._stemmer(text)

        text = [BOS] + text + [EOS]

        if 'weight' in item:

          if text_field_name not in self._weights:
            self._weights[text_field_name] = []
          weights = [float(weight) for weight in item['weight'].split()]
          # TODO un-hardcode
          # add 1.0 for BOS and EOS
          weights = [1.0] + weights + [1.0]
          self._weights[text_field_name].append(weights)

        if self._args['stopwords']:
          if 'weight' in item:
            text, self._weights[text_field_name][index] = \
              remove_stopwords(text, stopwords='nltk',
                               weights=self._weights[text_field_name][index])
          else:
            text = remove_stopwords(text, stopwords='nltk')

        if len(text) < 3:
          print(" Empty text in", self._json_dir, 'index', index)
        # assert len(text) >= 3, old_text

        if len(text) < min_seq_len:
          min_seq_len = len(text)

        # print('{}: {} ({})'.format(item['index'], text, text_field_name))
        self._sequences[text_field_name].append(text)

        # length of cleaned text (including BOS and EOS)
        self._sequence_lengths[text_field_name].append(len(text))

    print('Minimum sequence length: %d' % min_seq_len)

    for text_field_name in self._args['text_field_names']:
      # Check that every example has every field
      assert len(self._sequences[text_field_name]) == self._num_examples, \
        str(self._num_examples) + ' ' + str(
          len(self._sequences[text_field_name]))
      assert len(self._sequence_lengths[text_field_name]) == self._num_examples

  def get_tokenizer(self):
    if self._args['tokenizer_'] == "tweet_tokenizer":
      self._tokenizer = tweet_tokenizer.tokenize
    elif self._args['tokenizer_'] == "tweet_tokenizer_keep_handles":
      self._tokenizer = tweet_tokenizer_keep_handles.tokenize
    elif self._args['tokenizer_'] == "ruder_tokenizer":
      self._tokenizer = functools.partial(ruder_tokenizer, preserve_case=False)
    elif self._args['tokenizer_'] == "split_tokenizer":
      self._tokenizer = functools.partial(split_tokenizer)
    elif self._args['tokenizer_'] == "lower_tokenizer":
      self._tokenizer = functools.partial(lower_tokenizer)
    else:
      raise ValueError("unrecognized tokenizer: %s" % self._args['tokenizer_'])

  def get_stemmer(self):
    if self._args['stemmer'] == 'porter_stemmer':
      self._stemmer = porter_stemmer
    elif self._args['stemmer'] == 'snowball_stemmer':
      self._stemmer = snowball_stemmer
    elif self._args['stemmer'] == 'wordnet_stemmer':
      self._stemmer = wordnet_stemmer
    elif not self._args['stemmer']:
      self._stemmer = None
    else:
      raise ValueError("unrecognized stemmer: %s" % self._args['stemmer'])

  def build_vocab(self):
    """Builds vocabulary for this dataset only using tensorflow's

    VocabularyProcessor
    This vocabulary is only used for this dataset('s training data)
    """
    self.init_vocab_processor()

    training_docs = self.get_training_docs()

    self._vocab_processor.fit(training_docs)

    self.transform_text()

    self._vocab_freq_dict = self._vocab_processor.vocabulary_.freq

    return self._vocab_processor.vocabulary_

  def save_vocab_extra(self):
    make_dir(self._save_vocab_dir)
    with codecs.open(os.path.join(self._tfrecord_dir,
                                  "vocab_extra_v2i.json"),
                     mode='w', encoding='utf-8') as file:
      json.dump(self._vocab_extra, file,
                ensure_ascii=False, indent=4)

  def save_vocab_freq(self):
    make_dir(self._save_vocab_dir)
    with codecs.open(os.path.join(self._save_vocab_dir, "vocab_freq.json"),
                     mode='w', encoding='utf-8')as file:
      json.dump(self._vocab_freq_dict, file,
                ensure_ascii=False, indent=4)

  def save_v2i_dict(self):
    make_dir(self._save_vocab_dir)
    with codecs.open(os.path.join(self._save_vocab_dir, "vocab_v2i.json"),
                     mode='w', encoding='utf-8')as file:
      # TODO ???
      if self._categorical_vocab:
        json.dump(self._categorical_vocab.mapping, file,
                  ensure_ascii=False, indent=4)
      else:
        json.dump(self._vocab_v2i_dict, file,
                  ensure_ascii=False, indent=4)

  def save_i2v_dict(self):
    make_dir(self._save_vocab_dir)
    vocab_i2v_dict = dict()
    for i in range(len(self._categorical_vocab.reverse_mapping)):
      vocab_i2v_dict[i] = self._categorical_vocab.reverse_mapping[i]
    with codecs.open(os.path.join(self._save_vocab_dir, "vocab_i2v.json"),
                     mode='w', encoding='utf-8')as file:
      json.dump(vocab_i2v_dict, file, ensure_ascii=False, indent=4)

  def save_vocab(self):

    # TODO vocab_dir  vs save_vocab_dir

    # save the built vocab to the disk for future use
    if not self._save_vocab_dir:
      self._save_vocab_dir = self._tfrecord_dir
    make_dir(self._save_vocab_dir)

    if self._vocab_freq_dict:
      self.save_vocab_freq()
    if self._vocab_v2i_dict or self._categorical_vocab:
      self.save_v2i_dict()
    if self._vocab_i2v_dict:
      self.save_i2v_dict()

  def get_training_docs(self):
    # build vocabulary only according to training data
    training_docs = [self._sequences[text_field_name][i]
                     for text_field_name in self._args['text_field_names']
                     for i in self._train_index]
    if self._args['vocab_all']:
      training_docs += [self._sequences[text_field_name][i]
                        for text_field_name in self._args['text_field_names']
                        for i in self._valid_index]
      training_docs += [self._sequences[text_field_name][i]
                        for text_field_name in self._args['text_field_names']
                        for i in self._test_index]
    return training_docs

  def build_save_basic_vocab(self):
    """Build vocabulary with min_frequency=0 for this dataset'

    training data only and save to the directory
    minimum frequency is always 0 so that all the words of this dataset(
    's training data) are taken into account when merging with other
    vocabularies"""

    self._vocab_processor = VocabularyProcessor(
      max_document_length=self._args['max_document_length'],
      tokenizer_fn=tokenizer_simple)

    # build vocabulary only according to training data
    training_docs = self.get_training_docs()

    self._vocab_processor.fit(training_docs)
    self._categorical_vocab = self._vocab_processor.vocabulary_

    self._vocab_freq_dict = self._vocab_processor.vocabulary_.freq
    print("total word size =", len(self._vocab_freq_dict))

    self._save_vocab_dir = self._vocab_dir
    self.save_vocab_freq()

  def load_make_vocab(self):
    """Load word frequency vocabulary and generate word id mapping"""
    make_dir(self._vocab_dir)
    print('Use word frequency dictionary:',
          os.path.join(self._vocab_dir, 'vocab_freq.json'))
    with codecs.open(os.path.join(self._vocab_dir, 'vocab_freq.json'),
                     mode='r', encoding='utf-8') as file:
      self._vocab_freq_dict = json.load(file)

    categorical_vocab = CategoricalVocabulary(unknown_token=OOV)
    for word in self._vocab_freq_dict:
      categorical_vocab.add(word, count=self._vocab_freq_dict[word])
    categorical_vocab.trim(min_frequency=self._args['min_frequency'],
                           max_frequency=self._args['max_frequency'],
                           max_vocab_size=self._args['max_vocab_size'])
    categorical_vocab.freeze()
    return categorical_vocab

  def init_vocab_processor(self):
    self._vocab_processor = VocabularyProcessor(
      max_document_length=self._args['max_document_length'],
      max_vocab_size=self._args['max_vocab_size'],
      min_frequency=self._args['min_frequency'],
      max_frequency=self._args['max_frequency'],
      tokenizer_fn=tokenizer_simple)

  def get_train_vocab_list(self):
    """Get all the word types in the training docs

    :param doc_list: a list of documents
    :return: list, all the word types in the docs; use list to keep order
    """
    self.init_vocab_processor()

    training_docs = self.get_training_docs()

    self._vocab_processor.fit(training_docs)

    return self._vocab_processor.vocabulary_.reverse_mapping

  def load_vocab(self):
    make_dir(self._vocab_dir)

    if self._load_vocab_name == 'vocab_freq.json':
      # used when to merge new vocabulary using the vocabulary given

      print('Use word frequency dictionary:',
            os.path.join(self._vocab_dir, 'vocab_freq.json'))

      with codecs.open(os.path.join(self._vocab_dir, 'vocab_freq.json'),
                       mode='r', encoding='utf-8') as file:
        self._vocab_freq_dict = json.load(file)

      categorical_vocab = CategoricalVocabulary(unknown_token=OOV)
      for word in self._vocab_freq_dict:
        categorical_vocab.add(word, count=self._vocab_freq_dict[word])
      categorical_vocab.trim(min_frequency=self._args['min_frequency'],
                             max_frequency=self._args['max_frequency'],
                             max_vocab_size=self._args['max_vocab_size'])
      categorical_vocab.freeze()
      self._categorical_vocab = categorical_vocab
      # TODO bug ?
      self.init_vocab_processor()


    else:
      # used when to directly use the vocabulary given
      # e.g. in predict/test extra mode or use pretrained word embeddings
      if self._load_vocab_name == 'vocab_v2i.json':
        print('Use self-generated vocab mapping.')
        # this vocabulary mapping is generated solely on the training data
        with codecs.open(os.path.join(self._vocab_dir, 'vocab_v2i.json'),
                         mode='r', encoding='utf-8') as file:
          self._vocab_v2i_dict = json.load(file)
      else:
        # use the pretrained word embeddings' dictionary

        train_vocab_list = self.get_train_vocab_list()
        pretrained_path = os.path.join(self._vocab_dir,
                                       self._load_vocab_name)

        if self._args['pretrained_only']:
          # Using the pre-trained word embeddings' vocab only
          # Converting everything else into OOV
          # TODO
          self._vocab_v2i_dict = load_pretrianed_vocab_dict(pretrained_path)
        else:
          # Vocab must contain all training data's vocab
          if self._args['expand_vocab']:
            print('Combine pre-trained word embeddings\' vocabulary mapping '
                  'with all the word types appearing in the training data.')
            # TODO multiple training data for merged vocabulary
            # raise NotImplementedError('Combine pre-trained word embedding '
            #                           'and training data dictionary Not '
            #                           'Implemented!')
            # vocab_all = union(vocab_pretrained, vocab_train)
            # train_vocab_list = self.get_train_vocab_list()

            self._vocab_v2i_dict, self._vocab_extra = combine_vocab(
              pretrained_path,
              train_vocab_list)

            self._args['vocab_size'] = len(self._vocab_v2i_dict)

            # save the combined vocab to the disk for future use
            self._save_vocab_dir = self._tfrecord_dir
            self.save_v2i_dict()
            self.save_vocab_extra()
          else:
            # use pre-trained word embeddings' dictionary + EOS + OOV + LINEBREAK
            # print('Use pre-trained word embeddings\' vocabulary mapping only.')
            # # use the pretrained word embeddings' dictionary solely
            # glove_embedding = glove.Glove.load_stanford(os.path.join(
            #   self._vocab_dir, self._load_vocab_name))
            # reverse_mapping = ['<UNK>', '<EOS>', 'LINEBREAK'] + list(
            #   glove_embedding.dictionary)
            # self._vocab_v2i_dict ={w: i for i, w in enumerate(reverse_mapping)}

            # use training vocab only, reorder vocab to [not in glove, in glove]
            print('Use training vocab only.')

            self._args['random_size'], self._vocab_v2i_dict = reorder_vocab(
              pretrained_path,
              train_vocab_list)

            self._args['vocab_size'] = len(self._vocab_v2i_dict)

            # save the new vocab to the disk for future use
            make_dir(self._tfrecord_dir)
            self._args['reverse_vocab_path'] = os.path.join(self._tfrecord_dir,
                                                            "vocab_i2v.json")
            with codecs.open(self._args['reverse_vocab_path'], mode='w',
                             encoding='utf-8') as file:
              json.dump(self._vocab_v2i_dict, file,
                        ensure_ascii=False, indent=4)

      # build vocabulary processor using the loaded mapping
      categorical_vocab = CategoricalVocabulary(
        unknown_token=OOV, mapping=self._vocab_v2i_dict)
      self._vocab_processor = VocabularyProcessor(
        vocabulary=categorical_vocab,
        max_document_length=self._args['max_document_length'],
        tokenizer_fn=tokenizer_simple)
      assert categorical_vocab.mapping == self._vocab_v2i_dict

    # save the vocab if using pre-trained word embeddings
    # TODO ???
    if not self._load_vocab_name == 'vocab_v2i.json':
      self._categorical_vocab = categorical_vocab
      self._save_vocab_dir = self._tfrecord_dir
      self.save_vocab()

    self.transform_text()

    return self._vocab_processor.vocabulary_

  def transform_text(self):
    if self._args['padding']:
      # TODO: update implementation of transform_pad() to take a max length
      # as different kinds of sequences within a single example will have
      # different max lengths
      for text_field_name in self._args['text_field_names']:
        self._sequences[text_field_name] = list(
          self._vocab_processor.transform_pad(
            self._sequences[text_field_name]))
    else:
      for text_field_name in self._args['text_field_names']:
        self._sequences[text_field_name] = list(
          self._vocab_processor.transform(self._sequences[text_field_name]))

    for text_field_name in self._args['text_field_names']:
      self._sequences[text_field_name] = [list(i)
                                          for i
                                          in self._sequences[text_field_name]]

  def write_examples(self, file_name, split_index, labeled):
    # write to TFRecord data file
    tf.logging.info("Writing to: %s", file_name)
    with tf.python_io.TFRecordWriter(file_name) as writer:
      for index in tqdm(split_index):
        feature = dict()

        # Gather sequences and sequence statistics
        feature['index'] = tf.train.Feature(
          int64_list=tf.train.Int64List(
            value=[index]))

        if self._ids:
          feature['id'] = tf.train.Feature(
            bytes_list=tf.train.BytesList(
              value=[str(self._ids[index]).encode('utf-8')]))

        for text_field_name in self._args['text_field_names']:
          feature[text_field_name] = tf.train.Feature(
            int64_list=tf.train.Int64List(
              value=self._sequences[text_field_name][index]))
          feature[text_field_name + '_length'] = tf.train.Feature(
            int64_list=tf.train.Int64List(
              value=[self._sequence_lengths[text_field_name][index]]))

          if self._weights:
            feature[text_field_name + '_weights'] = tf.train.Feature(
              float_list=tf.train.FloatList(
                value=self._weights[text_field_name][index])
            )

          types, counts = get_types_and_counts(
            self._sequences[text_field_name][index])  # including BOS and EOS
          assert len(types) == len(counts)
          assert len(types) > 0
          for t in types:
            assert t >= 0
            assert t < self._args['vocab_size']
          for c in counts:
            assert c > 0
            assert c <= len(self._sequences[text_field_name][index])

          feature[text_field_name + '_types'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=types))
          feature[text_field_name + '_type_counts'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=counts))
          feature[text_field_name + '_types_length'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[len(types)]))

          if self._args['write_bow']:
            # This assumes a single vocabulary shared among all sequence kinds
            bow = bag_of_words(self._sequences[text_field_name][index],
                               self._args['vocab_size']).tolist()
            feature[text_field_name + '_bow'] = tf.train.Feature(
              float_list=tf.train.FloatList(value=bow))

          # if self._args['write_unique']:
          #   feature[text_field_name + '_unique'] = tf.train.Feature(
          #     int64_list=tf.train.Int64List(
          #       value=list(OrderedDict.fromkeys(
          #         self._sequences[text_field_name][index]))))  # keep order

        # Gather label

        if labeled:
          label = self._label_list[index]
          assert label is not None

          if self._args['label_type'] == 'int':
            feature['label'] = tf.train.Feature(
              int64_list=tf.train.Int64List(
                value=[label]))
          elif self._args['label_type'] == 'float':
            feature['label'] = tf.train.Feature(
              float_list=tf.train.FloatList(
                value=[label]))
          else:
            raise TypeError('Label type other than "int" or "float" not '
                            'implemented!')
        else:
          # label = self._label_list[index]
          # assert label is None
          pass

        if self._args['write_tfidf']:
          raise NotImplementedError("tfidf not supported")
          # feature['tfidf'] = tf.train.Feature(
          #   float_list=tf.train.FloatList(
          #     value=self._tfidf_list[index]
          #   )
          # )

        example = tf.train.Example(
          features=tf.train.Features(
            feature=feature))
        writer.write(example.SerializeToString())

  def split(self, index_path, train_ratio, valid_ratio, random_seed,
            subsample_ratio):
    """
    return given/randomly generated train/valid/test/unlabeled
        split indices
    """
    if not Path(index_path).exists():
      # no split given
      print("no split given")
      train_ind, valid_ind, test_ind = self.random_split_train_valid_test(
        self._num_examples,
        train_ratio, valid_ratio,
        random_seed)
      unlabeled_ind = []
    else:
      with gzip.open(index_path, mode='rt') as file:
        index_dict = json.load(file, encoding='utf-8')
      assert 'train' in index_dict and 'test' in index_dict
      train_ind = index_dict['train']
      test_ind = index_dict['test']
      if 'valid' in index_dict:
        print("Train/valid/test splits given. Use the default split.")
        valid_ind = index_dict['valid']
      else:
        print("Train/test splits given. Split train into train/valid.")
        train_ind, valid_ind = self.random_split_train_valid(
          train_ind, valid_ratio, random_seed)
      if 'unlabeled' in index_dict:
        print("This dataset has unlabeled data.")
        unlabeled_ind = index_dict['unlabeled']
      else:
        print("This dataset doesn't have unlabeled data.")
        unlabeled_ind = []

    train_ind_set = set(train_ind)
    valid_ind_set = set(valid_ind)
    test_ind_set = set(test_ind)
    unlabeled_ind_set = set(unlabeled_ind)

    print("Checking index duplications...")
    assert len(train_ind) == len(train_ind_set)
    assert len(valid_ind) == len(valid_ind_set)
    assert len(test_ind) == len(test_ind_set)
    assert len(unlabeled_ind) == len(unlabeled_ind_set)

    print("Checking index intersections...")
    assert len(train_ind_set.intersection(valid_ind_set)) == 0
    assert len(train_ind_set.intersection(test_ind_set)) == 0
    assert len(train_ind_set.intersection(unlabeled_ind_set)) == 0
    assert len(valid_ind_set.intersection(test_ind_set)) == 0
    assert len(valid_ind_set.intersection(unlabeled_ind_set)) == 0
    assert len(test_ind_set.intersection(unlabeled_ind_set)) == 0

    print("Before subsampling: train : valid : test : unlabeled = %d : %d : "
          "%d : %d" %
          (len(train_ind),
           len(valid_ind),
           len(test_ind),
           len(unlabeled_ind)))

    if subsample_ratio is not None and subsample_ratio < 1.0:
      train_ind = self.subsample(
        train_ind, random_seed, subsample_ratio)
      valid_ind = self.subsample(
        valid_ind, random_seed, subsample_ratio)
      test_ind = self.subsample(
        test_ind, random_seed, subsample_ratio)
      unlabeled_ind = self.subsample(
        unlabeled_ind, random_seed, subsample_ratio)

      print("After subsampling, train : valid : test : unlabeled = %d : %d : "
            "%d : %d" %
            (len(train_ind),
             len(valid_ind),
             len(test_ind),
             len(unlabeled_ind)))
    else:
      print('No subsampling.')

    return train_ind, valid_ind, test_ind, unlabeled_ind

  def write_args(self):
    # save dataset arguments

    self._args['has_ids'] = self._ids is not None
    self._args['has_weights'] = self._weights is not None
    self._args['max_vocab_size_allowed'] = self._args.pop('max_vocab_size')

    print('Arguments for the dataset:')
    for k, v in self._args.items():
      print(k, ':', v)

    args_path = os.path.join(self._tfrecord_dir, "args.json")
    with codecs.open(args_path, mode='w', encoding='utf-8') as file:
      json.dump(self._args, file, ensure_ascii=False, indent=4)
    assert os.path.exists(args_path), 'Failed to save args.json!'

  @staticmethod
  def subsample(index, random_seed, subsample_ratio=0.1):
    np.random.seed(random_seed)
    index = np.random.permutation(index)
    return np.split(index, [int(subsample_ratio * len(index))])[0]

  @staticmethod
  def random_split_train_valid_test(length, train_ratio, valid_ratio,
                                    random_seed):
    index = np.array(list(xrange(length)))
    np.random.seed(random_seed)
    index = np.random.permutation(index)

    return np.split(index,
                    [int(train_ratio * len(index)),
                     int((train_ratio + valid_ratio) * len(index))])

  @staticmethod
  def random_split_train_valid(train_index, valid_ratio, random_seed):
    """Takes part of training data to validation data"""
    index = np.array(train_index)
    np.random.seed(random_seed)
    index = np.random.permutation(index)
    return np.split(index, [int((1.0 - valid_ratio) * len(index))])

  @property
  def args(self):
    return self._args

  @property
  def vocab_size(self):
    return self._args['vocab_size']

  @property
  def max_document_length(self):
    return self._args['max_document_length']

  @property
  def mapping(self):
    return self._categorical_vocab.mapping

  @property
  def reverse_mapping(self):
    return self._categorical_vocab.reverse_mapping


def merge_save_vocab_dicts(vocab_paths, save_path):
  """
  :param vocab_paths: list of vocabulary paths
  :param save_path: path to save the merged vocab
  :return:
  """
  merged_vocab_dict = dict()
  for path in vocab_paths:
    with codecs.open(path, mode='r', encoding='utf-8') as file:
      vocab_dict = json.load(file)
      merged_vocab_dict = combine_dicts(merged_vocab_dict, vocab_dict)

  # sort merged vocabulary according to frequency
  merged_vocab_list = sorted(merged_vocab_dict.items(),
                             key=operator.itemgetter(1),
                             reverse=True)
  merged_vocab_dict = dict()
  for i in merged_vocab_list:
    merged_vocab_dict[i[0]] = i[1]

  with codecs.open(save_path, mode='w', encoding='utf-8') as file:
    json.dump(merged_vocab_dict, file, ensure_ascii=False, indent=4)


def combine_dicts(x, y):
  return {i: x.get(i, 0) + y.get(i, 0) for i in
          set(itertools.chain(x, y))}


def merge_dict_write_tfrecord(json_dirs,
                              tfrecord_dirs,
                              merged_dir,
                              max_document_length=-1,
                              max_vocab_size=-1,
                              min_frequency=0,
                              max_frequency=-1,
                              text_field_names=['text'],
                              label_field_name='label',
                              label_type='int',
                              tokenizer_='tweet_tokenizer',
                              stemmer='porter_stemmer',
                              stopwords='nltk',
                              train_ratio=TRAIN_RATIO,
                              valid_ratio=VALID_RATIO,
                              subsample_ratio=1,
                              padding=False,
                              write_bow=False,
                              write_tfidf=False,
                              preproc=True,
                              vocab_all=False):
  """Merge all the dictionaries for each dataset and write TFRecord files

  1. generate word frequency dictionary for each dataset
  2. add them up to a new word frequency dictionary
  3. generate the word id mapping using arguments
  4. use the same word id mapping to generate TFRecord files for each dataset

  :param json_dirs: list of dataset(in json.gz) directories
  :param tfrecord_dirs: list of directories to save the TFRecord files
  :param merged_dir: new directory to save all the data
  :return: args_dicts: list of args(dict) of each dataset
  """

  # generate vocab for every dataset without writing their own TFRecord files
  # the generated vocab freq dicts shall be saved at
  # json_dir/vocab_freq_dict.json

  # Assumes that all datasets have
  # the same text_field_names and label_field_name TODO
  # max_document_lengths = []
  for json_dir, tfrecord_dir in zip(json_dirs, tfrecord_dirs):
    dataset = Dataset(json_dir,
                      tfrecord_dir=tfrecord_dir,
                      vocab_dir=merged_dir,
                      max_document_length=max_document_length,
                      max_vocab_size=max_vocab_size,
                      min_frequency=min_frequency,
                      max_frequency=-max_frequency,
                      text_field_names=text_field_names,
                      label_field_name=label_field_name,
                      label_type=label_type,
                      tokenizer_=tokenizer_,
                      stemmer=stemmer,
                      stopwords=stopwords,
                      generate_basic_vocab=True,
                      vocab_given=False,
                      generate_tf_record=False,
                      preproc=preproc,
                      vocab_all=vocab_all)
    # max_document_lengths.append(dataset.max_document_length)
  # if max_document_length == -1:
  #   max_document_length = max(max_document_lengths)

  # merge all the vocabularies
  vocab_paths = []
  for json_dir in json_dirs:
    vocab_path = os.path.join(json_dir, "vocab_freq.json")
    vocab_paths.append(vocab_path)
  merge_save_vocab_dicts(vocab_paths, os.path.join(merged_dir,
                                                   "vocab_freq.json"))

  print("merged public word frequency dictionary saved to path",
        os.path.join(merged_dir, "vocab_freq.json"))

  # generate word id mapping, which is then used as the merged vocabulary
  # for all the datasets
  dataset = Dataset(json_dir=None,
                    tfrecord_dir=None,  # TODO
                    vocab_dir=merged_dir,
                    generate_basic_vocab=False,
                    vocab_given=True,
                    vocab_name='vocab_freq.json',
                    generate_tf_record=False,
                    text_field_names=text_field_names,
                    label_field_name=label_field_name,
                    label_type=label_type,
                    max_document_length=max_document_length,
                    min_frequency=min_frequency,
                    max_frequency=max_frequency,
                    max_vocab_size=max_vocab_size,
                    # train_ratio=train_ratio,
                    # valid_ratio=valid_ratio,
                    # subsample_ratio=subsample_ratio,
                    # padding=padding,
                    # write_bow=write_bow,
                    # write_tfidf=write_tfidf,
                    preproc=preproc,
                    vocab_all=vocab_all
                    )
  with codecs.open(os.path.join(merged_dir, 'vocab_v2i.json'),
                   mode='w', encoding='utf-8') as file:
    json.dump(dataset.mapping, file, ensure_ascii=False, indent=4)

  vocab_i2v_dict = dict()
  for i in range(len(dataset.reverse_mapping)):
    vocab_i2v_dict[i] = dataset.reverse_mapping[i]
  with codecs.open(os.path.join(merged_dir, 'vocab_i2v.json'),
                   mode='w', encoding='utf-8') as file:
    json.dump(vocab_i2v_dict, file, ensure_ascii=False, indent=4)

  # with open(os.path.join(merged_dir, "vocab_size.txt"), "w") as file:
  #   file.write(str(dataset.vocab_size))

  # write TFRecords for each dataset with the same word id mapping
  args_dicts = []
  for json_dir in json_dirs:
    tfrecord_dir = os.path.join(merged_dir, os.path.basename(
      os.path.normpath(json_dir)))
    dataset = Dataset(json_dir,
                      tfrecord_dir=tfrecord_dir,
                      vocab_dir=merged_dir,
                      generate_basic_vocab=False,
                      vocab_given=True,
                      vocab_name='vocab_v2i.json',
                      generate_tf_record=True,
                      text_field_names=text_field_names,
                      label_field_name=label_field_name,
                      label_type=label_type,
                      max_document_length=max_document_length,
                      max_vocab_size=max_vocab_size,
                      min_frequency=min_frequency,
                      max_frequency=max_frequency,
                      train_ratio=train_ratio,
                      valid_ratio=valid_ratio,
                      subsample_ratio=subsample_ratio,
                      padding=padding,
                      write_bow=write_bow,
                      write_tfidf=write_tfidf,
                      tokenizer_=tokenizer_,
                      stemmer=stemmer,
                      stopwords='nltk',
                      preproc=preproc,
                      vocab_all=vocab_all
                      )
    args_dicts.append(dataset.args)

  return args_dicts


# TODO pretrained_only
def merge_pretrain_write_tfrecord(json_dirs,
                                  tfrecord_dirs,
                                  merged_dir,
                                  vocab_dir,
                                  vocab_name,
                                  min_frequency,
                                  max_frequency,
                                  max_vocab_size,
                                  max_document_length=-1,
                                  text_field_names=['text'],
                                  label_field_name='label',
                                  label_type='int',
                                  tokenizer_="tweet_tokenizer",
                                  stemmer=None,
                                  stopwords='nltk',
                                  train_ratio=TRAIN_RATIO,
                                  valid_ratio=VALID_RATIO,
                                  subsample_ratio=1,
                                  padding=False,
                                  write_bow=False,
                                  write_tfidf=False,
                                  expand_vocab=False,
                                  pretrained_only=False,
                                  preproc=True,
                                  vocab_all=True):
  """Use the dictionary of the pre-trained word embedding, combine the words

  from the training data of all the datasets if necessary

  1. get the vocabulary v2i mapping of the pre-trained word embedding
  2. get all the word types from the training data of all the datasets if
  necessary
  3. combine the vocabulary if necessary
  4. write TFRecord files for each dataset using the vocabulary

  :param json_dirs: list of dataset(in json.gz) directories
  :param tfrecord_dirs: list of directories to save the TFRecord files
  :param merged_dir: new directory to save all the data
  :return: args_dicts: list of args(dict) of each dataset
  """

  # generate vocab for every dataset without writing their own TFRecord files
  # the generated vocab freq dicts shall be saved at
  # json_dir/vocab_freq_dict.json

  pretrained_path = os.path.join(vocab_dir, vocab_name)

  # get the vocab from the training data of each dataset
  # Assumes that all datasets have
  # the same text_field_names and label_field_name
  train_vocab_set = set()  # use list to keep order
  if padding:
    max_document_lengths = []
  for json_dir, tfrecord_dir in zip(json_dirs, tfrecord_dirs):
    dataset = Dataset(json_dir,
                      tfrecord_dir=tfrecord_dir,
                      vocab_dir=merged_dir,
                      max_document_length=max_document_length,
                      max_vocab_size=max_vocab_size,
                      min_frequency=min_frequency,
                      max_frequency=max_frequency,
                      text_field_names=text_field_names,
                      label_field_name=label_field_name,
                      label_type=label_type,
                      tokenizer_=tokenizer_,
                      stemmer=stemmer,
                      stopwords=stopwords,
                      generate_basic_vocab=True,
                      vocab_given=False,
                      generate_tf_record=False,
                      preproc=preproc,
                      vocab_all=vocab_all)
    train_vocab_set = train_vocab_set.union(set(dataset.mapping))
    if padding:
      max_document_lengths.append(max_document_length)
  if padding:
    max_document_length = max(max_document_lengths)

  specials = [OOV, BOS, EOS]  # make sure OOV is indexed 0
  if LINEBREAK in train_vocab_set:
    specials += [LINEBREAK]  # linebreaks may not appear in some sentence-level
    #  dataset

  train_vocab_set.difference_update(set(specials))
  train_vocab_list = specials + list(train_vocab_set)

  if not expand_vocab:
    print('Use training vocab only.')

    random_size, vocab_v2i_all = reorder_vocab(pretrained_path,
                                               train_vocab_list)

    with open(os.path.join(merged_dir, "random_size.txt"),
              "w") as file:
      file.write(str(random_size))

  else:

    vocab_v2i_all, vocab_extra = combine_vocab(pretrained_path,
                                               train_vocab_list)

    with codecs.open(os.path.join(merged_dir,
                                  "vocab_extra_v2i.json"),
                     mode='w', encoding='utf-8') as file:
      json.dump(vocab_extra, file,
                ensure_ascii=False, indent=4)

  with codecs.open(os.path.join(merged_dir, 'vocab_v2i.json'),
                   mode='w', encoding='utf-8') as file:
    json.dump(vocab_v2i_all, file, ensure_ascii=False, indent=4)

  vocab_i2v_dict = dict()
  for v, i in vocab_v2i_all.items():
    vocab_i2v_dict[i] = v
  with codecs.open(os.path.join(merged_dir, 'vocab_i2v.json'),
                   mode='w', encoding='utf-8') as file:
    json.dump(vocab_i2v_dict, file, ensure_ascii=False, indent=4)

  # with open(os.path.join(merged_dir, 'vocab_size.txt'), 'w') as file:
  #   file.write(str(len(vocab_v2i_all)))

  # write TFRecords for each dataset with the same word id mapping
  args_dicts = []
  for json_dir in json_dirs:
    tfrecord_dir = os.path.join(merged_dir, os.path.basename(
      os.path.normpath(json_dir)))
    dataset = Dataset(json_dir,
                      tfrecord_dir=tfrecord_dir,
                      generate_basic_vocab=False,
                      vocab_given=True,
                      vocab_dir=merged_dir,
                      vocab_name='vocab_v2i.json',
                      generate_tf_record=True,
                      text_field_names=text_field_names,
                      label_field_name=label_field_name,
                      label_type=label_type,
                      max_document_length=max_document_length,
                      max_vocab_size=max_vocab_size,
                      min_frequency=min_frequency,
                      max_frequency=max_frequency,
                      train_ratio=train_ratio,
                      valid_ratio=valid_ratio,
                      subsample_ratio=subsample_ratio,
                      padding=padding,
                      write_bow=write_bow,
                      write_tfidf=write_tfidf,
                      tokenizer_=tokenizer_,
                      stemmer=stemmer,
                      stopwords='nltk',
                      preproc=preproc,
                      vocab_all=vocab_all)
    args_dicts.append(dataset.args)

  return args_dicts


def get_types_and_counts(token_list):
  counts = {x: token_list.count(x) for x in token_list}
  return counts.keys(), counts.values()


def main():
  # test
  json_dirs = ["./vocab_test/1/", "./vocab_test/2/", "./vocab_test/3/"]
  tfrecord_dirs = ['./vocab_test/1/min_0_max_0/',
                   './vocab_test/2/min_0_max_0/',
                   './vocab_test/3/min_0_max_0/']
  merge_dict_write_tfrecord(json_dirs, tfrecord_dirs,
                            merged_dir="./vocab_test/merged/",
                            write_bow=True, write_tfidf=True)


if __name__ == '__main__':
  main()

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
# ==============================================================================

# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import gzip
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tasks.code.data_prep import *
from tasks.code.text import VocabularyProcessor
from tasks.code.util import bag_of_words
from tensorflow.contrib.learn.python.learn.preprocessing import \
  CategoricalVocabulary
from tqdm import tqdm

flags = tf.flags
logging = tf.logging

FLAGS = flags.FLAGS
from six.moves import xrange

TRAIN_RATIO = 0.8  # train out of all
VALID_RATIO = 0.1  # valid out of all / valid out of train
RANDOM_SEED = 42


class Dataset():
  def __init__(self,
               json_dir,
               load_vocab,
               generate_basic_vocab,
               generate_tf_record,
               vocab_dir,
               tfrecord_dir,
               max_document_length=-1,
               min_frequency=0,
               max_frequency=-1,
               text_field_names=['text'],
               label_field_name='label',
               train_ratio=TRAIN_RATIO,
               valid_ratio=VALID_RATIO,
               random_seed=RANDOM_SEED,
               subsample_ratio=1,
               padding=False,
               write_bow=False
               ):
    """
    :param data_type: whether the data is json data or tf records
    :param json_dir: where data.json.gz and index.json.gz are
    located, and vocabulary/tf records built from the single datasets
    are to be saved
    :param load_vocab: True if to use the given vocabulary
    :param vocab_dir: directory to load the given (e.g. merged) vocabulary
    frequency dict json file or to save the self-generated vocabulary
    :param tfrecord_dir: directory to save generated TFRecord files
    :param max_document_length: maximum document length for the mapped
    word ids, computed as the maximum document length of all the
    training data if None
    :param min_frequency: minimum frequency to build the vocabulary,
    words that appear lower than or equal to this number would be discarded
    :param max_frequency: maximum frequency to build the vocabulray,
    words that appear more than or equal to this number would be discarded
    :param bow: True if to write bag of words as a feature in the tf record
    :param text_field_names: string of a list of text field names joined
    with spaces, read from json_dir if None
    :param label_field_name: label field name(only 1), read from
    json_dir if None
    :param valid_ratio: how many data out of all data to use as valid
    data if not splits are given, or how many data out of train data to
    use as valid if train/test splits are given
    :param train_ratio: how many data to use as train data if train
    split not given
    :param random_seed: random seed used in random spliting, makeing
    sure the same random split is used when given the same random seed
    :param subsample_ratio: randomly takes part of the datasets when it's
    too large
    :param generate_basic_vocab: True if the basic vocabulary(which
    shall be used to merge the public vocabulary) needs to be generated
    :param generate_tf_record: True if tf record files need generating
    :param padding: True if token(word id) list needs padding to max_document_length
    """

    print("data in", json_dir)

    with gzip.open(os.path.join(json_dir, "data.json.gz"), 'rt') as file:
      data = json.load(file, encoding='utf-8')
      file.close()
    self._label_list = [int(item[label_field_name]) if label_field_name in item else None for item in data]
    self._num_classes = len(set(self._label_list))

    # if sys.version_info[0] < 3:
    #     self.text_list = df[text_field_names].astype(unicode).sum(
    #         axis=1).tolist()
    # else:
    #     self.text_list = df[text_field_names].astype(str).sum(
    #         axis=1).tolist()

    self._text_list = [' '.join([item[text_field_name]])
                       for text_field_name in text_field_names for
                       item in data]

    self._text_list = [tweet_tokenizer.tokenize(text) + ['EOS'] for text in
                       self._text_list]
    self._token_length_list = [len(text) for text in
                               self._text_list]  # length of cleaned text (including EOS)

    self._padding = padding

    # tokenize and reconstruct as string(which vocabulary processor
    # takes as input)

    # get index
    print("Generating train/valid/test splits...")
    index_path = os.path.join(json_dir, "index.json.gz")
    self._train_index, self._valid_index, self._test_index, self._unlabeled_index \
      = self.split(index_path, train_ratio, valid_ratio, random_seed, subsample_ratio)

    # only compute from training data
    if max_document_length == -1:
      self._max_document_length = max(len(self._text_list[i]) for i in
                                      self._train_index)
      print("max document length (computed) =",
            self._max_document_length)
    else:
      self._max_document_length = max_document_length
      print("max document length (given) =", self._max_document_length)
    print(self._max_document_length)

    self._vocab_dict = None
    self._categorical_vocab = None

    # generate and save the vocabulary which contains all the words
    if generate_basic_vocab:
      print("Generating the basic vocabulary.")
      self.build_save_basic_vocab(vocab_dir=json_dir)

    if generate_tf_record is False:
      print("No need to generate tf records. Done. ")
      return

    if load_vocab is False:
      print("No vocabulary given. Generate a new one.")
      self._categorical_vocab = self.build_vocab(
        min_frequency=min_frequency,
        max_frequency=max_frequency)
      self.save_vocab(tfrecord_dir)

    else:
      print("Public vocabulary given. Use that to build vocabulary "
            "processor.")
      self._categorical_vocab = self.load_vocab(vocab_dir,
                                                min_frequency=min_frequency,
                                                max_frequency=max_frequency)
    # save mapping/reverse mapping to the disk
    # freq:            vocab_dir/vocab_freq.json
    # mapping:         vocab_dir/vocab_v2i.json
    # reverse mapping: vocab_dir/vocab_i2v.json(sorted according to freq)

    self._vocab_size = len(self._categorical_vocab._mapping)
    print("used vocab size =", self._vocab_size)

    # write to tf records
    try:
      os.stat(tfrecord_dir)
    except:
      os.mkdir(tfrecord_dir)
    self._train_path = os.path.join(tfrecord_dir, 'train.tf')
    self._valid_path = os.path.join(tfrecord_dir, 'valid.tf')
    self._test_path = os.path.join(tfrecord_dir, 'test.tf')

    print("Writing TFRecord files for the training data...")
    self.write_examples(self._train_path, self._train_index, write_bow, labeled=True)
    print("Writing TFRecord files for the validation data...")
    self.write_examples(self._valid_path, self._valid_index, write_bow, labeled=True)
    print("Writing TFRecord files for the test data...")
    self.write_examples(self._test_path, self._test_index, write_bow, labeled=True)

    # write unlabeled data to TFRecord files if there're any

    if len(self._unlabeled_index) == 0:
      print("Unlabeled data not found.")
      self._unlabeled_path = None
      self._has_unlabeled = False
    else:
      print("Unlabeled data found.")
      self._has_unlabeled = True
      print("Writing TFRecord files for the unlabeled data...")
      self._unlabeled_path = os.path.join(tfrecord_dir, 'unlabeled.tf')
      self.write_examples(self._unlabeled_path, self._unlabeled_index, write_bow, labeled=False)

    # save dataset arguments
    self._args = {
      'num_classes': self._num_classes,
      'max_document_length': self._max_document_length,
      'vocab_size': self._vocab_size,
      'min_frequency': min_frequency,
      'max_frequency': max_frequency,
      'random_seed': random_seed,
      'train_path': os.path.abspath(self._train_path),
      'valid_path': os.path.abspath(self._valid_path),
      'test_path': os.path.abspath(self._test_path),
      'train_size': len(self._train_index),
      'valid_size': len(self._valid_index),
      'test_size': len(self._test_index),
      'has_unlabeled': self._has_unlabeled,
      'unlabeled_path': os.path.abspath(self._unlabeled_path) if self._unlabeled_path is not None else None,
      'unlabeled_size': len(self._unlabeled_index)
    }
    print(self._args)
    args_path = os.path.join(tfrecord_dir, "args.json")
    with codecs.open(args_path, mode='w', encoding='utf-8') as file:
      json.dump(self._args, file, ensure_ascii=False, indent=4)
      file.close()

  def build_vocab(self, min_frequency, max_frequency):
    """Builds vocabulary for this dataset only using tensorflow's
    VocabularyProcessor

    This vocabulary is only used for this dataset('s training data)
    """
    vocab_processor = VocabularyProcessor(
      max_document_length=self._max_document_length,
      min_frequency=min_frequency,
      max_frequency=max_frequency,
      tokenizer_fn=tokenizer)

    # build vocabulary only according to training data
    vocab_processor.fit([self._text_list[i] for i in self._train_index])

    if self._padding is True:
      self._token_list = list(
        vocab_processor.transform_pad(self._text_list))
    else:
      self._token_list = list(
        vocab_processor.transform(self._text_list))
    self._token_list = [list(i) for i in self._token_list]
    self._vocab_freq_dict = vocab_processor.vocabulary_._freq

    return vocab_processor.vocabulary_

  def save_vocab(self, vocab_dir):

    # save the built vocab to the disk for future use
    try:
      os.stat(vocab_dir)
    except:
      os.mkdir(vocab_dir)

    with codecs.open(os.path.join(vocab_dir, "vocab_freq.json"),
                     mode='w', encoding='utf-8')as file:
      json.dump(self._vocab_freq_dict, file, ensure_ascii=False, indent=4)
      file.close()

    with codecs.open(os.path.join(vocab_dir, "vocab_v2i.json"),
                     mode='w', encoding='utf-8')as file:
      json.dump(self._categorical_vocab._mapping, file,
                ensure_ascii=False, indent=4)
      file.close()

    vocab_i2v_dict = dict()
    for i in range(len(self._categorical_vocab._reverse_mapping)):
      vocab_i2v_dict[i] = self._categorical_vocab._reverse_mapping[i]
    with codecs.open(os.path.join(vocab_dir, "vocab_i2v.json"),
                     mode='w', encoding='utf-8')as file:
      json.dump(vocab_i2v_dict, file, ensure_ascii=False, indent=4)
      file.close()

  def build_save_basic_vocab(self, vocab_dir):
    """Bulid vocabulary with min_frequency=0 for this dataset'

    training data only and save to the directory
    minimum frequency is always 0 so that all the words of this dataset(
    's training data) are taken into account when merging with other
    vocabularies"""

    vocab_processor = VocabularyProcessor(
      max_document_length=self._max_document_length,
      tokenizer_fn=tokenizer)

    # build vocabulary only according to training data
    vocab_processor.fit([self._text_list[i] for i in self._train_index])

    vocab_freq_dict = vocab_processor.vocabulary_._freq
    print("total word size =", len(vocab_freq_dict))
    try:
      os.stat(vocab_dir)
    except:
      os.mkdir(vocab_dir)

    with codecs.open(os.path.join(vocab_dir, "vocab_freq.json"),
                     mode='w', encoding='utf-8') as file:
      json.dump(vocab_freq_dict, file, ensure_ascii=False, indent=4)
      file.close()

  def load_vocab(self, vocab_dir, min_frequency, max_frequency):
    with codecs.open(os.path.join(vocab_dir, "vocab_freq.json"),
                     mode='r', encoding='utf-8') as file:
      self._vocab_freq_dict = json.load(file)
      file.close()
    categorical_vocab = CategoricalVocabulary()
    for word in self._vocab_freq_dict:
      categorical_vocab.add(word, count=self._vocab_freq_dict[word])
    categorical_vocab.trim(min_frequency=min_frequency,
                           max_frequency=max_frequency)
    categorical_vocab.freeze()

    vocab_processor = VocabularyProcessor(
      vocabulary=categorical_vocab,
      max_document_length=self._max_document_length,
      min_frequency=min_frequency,
      tokenizer_fn=tokenizer)

    if self._padding is True:
      self._token_list = list(
        vocab_processor.transform_pad(self._text_list))
    else:
      self._token_list = list(
        vocab_processor.transform(self._text_list))
    self._token_list = [list(i) for i in self._token_list]
    return vocab_processor.vocabulary_

  def write_examples(self, file_name, split_index, write_bow, labeled):
    # write to TFRecord data file
    tf.logging.info("Writing to: %s", file_name)
    with tf.python_io.TFRecordWriter(file_name) as writer:
      for index in tqdm(split_index):
        feature = {
          'tokens': tf.train.Feature(
            int64_list=tf.train.Int64List(
              value=self._token_list[index])),
          'tokens_length': tf.train.Feature(
            int64_list=tf.train.Int64List(
              value=[self._token_length_list[index]]))
        }
        if labeled:
          label = self._label_list[index]
          assert label is not None
          feature['label'] = tf.train.Feature(
            int64_list=tf.train.Int64List(
              value=[label]))
        else:
          label = self._label_list[index]
          assert label is None

        types, counts = get_types_and_counts(self._token_list[
                                               index])  # including EOS

        assert len(types) == len(counts)
        assert len(types) > 0

        for t in types:
          assert t >= 0
          assert t < self._vocab_size
        for c in counts:
          assert c > 0
          assert c <= len(self._token_list[index])

        feature['types'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=types))
        feature['type_counts'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=counts))
        feature['types_length'] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[len(types)]))

        if write_bow is True:
          feature['bow'] = tf.train.Feature(
            float_list=tf.train.FloatList(
              value=bag_of_words(
                self._token_list[index],
                self._vocab_size).tolist()))

        example = tf.train.Example(
          features=tf.train.Features(
            feature=feature))
        writer.write(example.SerializeToString())

  def split(self, index_path, train_ratio, valid_ratio, random_seed,
            subsample_ratio):
    """
    return given/randomly generated train/valid/test/unlabeled split indices
    """
    if not Path(index_path).exists():
      # no split given
      print("no split given")
      train_index, valid_index, test_index \
        = self.random_split_train_valid_test(len(self._text_list),
                                             train_ratio, valid_ratio,
                                             random_seed)
      unlabeled_index = []
    else:
      with gzip.open(index_path, mode='rt') as file:
        index_dict = json.load(file, encoding='utf-8')
        file.close()
      assert 'train' in index_dict and 'test' in index_dict
      train_index = index_dict['train']
      test_index = index_dict['test']
      if 'valid' in index_dict:
        print("train/valid/test splits given")
        valid_index = index_dict['valid']
      else:
        print("train/test splits given")
        train_index, valid_index = self.random_split_train_valid(
          train_index, valid_ratio, random_seed)
      if 'unlabeled' in index_dict:
        print("This dataset has unlabeled data.")
        unlabeled_index = index_dict['unlabeled']
      else:
        print("This dataset doesn't have unlabeled data.")
        unlabeled_index = []

    # no intersection
    assert (len(train_index) == len(set(train_index)))
    assert (len(valid_index) == len(set(valid_index)))
    assert (len(test_index) == len(set(test_index)))
    assert (len(unlabeled_index) == len(set(unlabeled_index)))

    assert len([i for i in train_index if i in valid_index]) == 0
    assert len([i for i in train_index if i in test_index]) == 0
    assert len([i for i in valid_index if i in test_index]) == 0
    assert len([i for i in train_index if i in unlabeled_index]) == 0
    assert len([i for i in valid_index if i in unlabeled_index]) == 0
    assert len([i for i in test_index if i in unlabeled_index]) == 0

    print("train : valid : test : unlabeled = %d : %d : %d : %d" %
          (len(train_index),
           len(valid_index),
           len(test_index),
           len(unlabeled_index)))

    if subsample_ratio is not None and subsample_ratio < 1.0:
      train_index = self.subsample(train_index, random_seed, subsample_ratio)
      valid_index = self.subsample(valid_index, random_seed, subsample_ratio)
      test_index = self.subsample(test_index, random_seed, subsample_ratio)
      unlabeled_index = self.subsample(unlabeled_index, random_seed, subsample_ratio)

      print("train : valid : test : unlabeled = %d : %d : %d : %d" %
            (len(train_index),
             len(valid_index),
             len(test_index),
             len(unlabeled_index)))

    return train_index, valid_index, test_index, unlabeled_index

  def subsample(self, index, random_seed, subsample_ratio=0.1):
    np.random.seed(random_seed)
    index = np.random.permutation(index)
    return np.split(index, [int(subsample_ratio * len(index))])[0]

  def random_split_train_valid_test(self, length, train_ratio, valid_ratio,
                                    random_seed):
    index = np.array(list(xrange(length)))
    np.random.seed(random_seed)
    index = np.random.permutation(index)

    return np.split(index,
                    [int(train_ratio * len(index)),
                     int((train_ratio + valid_ratio) * len(index))])

  def random_split_train_valid(self, train_index, valid_ratio, random_seed):
    """Takes part of training data to validation data"""
    index = np.array(train_index)
    np.random.seed(random_seed)
    index = np.random.permutation(index)
    return np.split(index, [int((1.0 - valid_ratio) * len(index))])

  @property
  def args(self):
    return self._args


def merge_save_vocab_dicts(vocab_paths, save_path):
  """
  :param vocab_paths: list of vocabulary paths
  :return:
  """
  merged_vocab_dict = dict()
  for path in vocab_paths:
    with codecs.open(path, mode='r', encoding='utf-8') as file:
      vocab_dict = json.load(file)
      file.close()
      merged_vocab_dict = combine_dicts(merged_vocab_dict, vocab_dict)

  with codecs.open(save_path, mode='w', encoding='utf-8') as file:
    json.dump(merged_vocab_dict, file, ensure_ascii=False, indent=4)
    file.close()


def combine_dicts(x, y):
  return {i: x.get(i, 0) + y.get(i, 0) for i in
          set(itertools.chain(x, y))}


def merge_dict_write_tfrecord(json_dirs, tfrecord_dirs, merged_dir,
                              max_document_length=-1, min_frequency=0,
                              max_frequency=-1, train_ratio=TRAIN_RATIO,
                              valid_ratio=VALID_RATIO, subsample_ratio=1,
                              padding=False, write_bow=False):
  """
  1. generate and save vocab dictionary which contains all the words(
  cleaned) for each dataset
  2. merge the vocabulary
  3. generate and save TFRecord files for each dataset using the merged vocab
  :param json_dirs: list of dataset directories
  :param json_dirs: new directory to save all the data
  :return: args_dicts: list of args(dict) of each dataset
  """
  # generate vocab for every dataset without writing their own TFRecord files
  # the generated vocab freq dicts shall be saved at
  # json_dir/vocab_freq_dict.json
  max_document_lengths = []
  for json_dir, tfrecord_dir in zip(json_dirs, tfrecord_dirs):
    dataset = Dataset(json_dir, tfrecord_dir=tfrecord_dir,
                      vocab_dir=merged_dir,
                      max_document_length=-1,
                      min_frequency=0,
                      max_frequency=-1,
                      generate_basic_vocab=True,
                      load_vocab=False,
                      generate_tf_record=False)
    max_document_lengths.append(dataset._max_document_length)

  # new data dir based all the datasets' names
  data_names = [os.path.basename(os.path.normpath(json_dir)) for json_dir
                in json_dirs]
  data_names.sort()
  try:
    os.stat(merged_dir)
  except:
    os.mkdir(merged_dir)

  # merge all the vocabularies
  vocab_paths = []
  for json_dir in json_dirs:
    vocab_path = os.path.join(json_dir, "vocab_freq.json")
    vocab_paths.append(vocab_path)
  merge_save_vocab_dicts(vocab_paths, os.path.join(merged_dir,
                                                   "vocab_freq.json"))

  print("merged public vocabulary saved to path", os.path.join(merged_dir,
                                                               "vocab_freq.json"))

  # write tf records
  vocab_i2v_lists = []
  vocab_v2i_dicts = []
  vocab_sizes = []
  args_dicts = []

  # max_document_length is only useful when padding is True
  if max_document_length is None:
    max_document_length = max(max_document_lengths)
  for json_dir in json_dirs:
    tfrecord_dir = os.path.join(merged_dir, os.path.basename(
      os.path.normpath(json_dir)))
    dataset = Dataset(json_dir,
                      tfrecord_dir=tfrecord_dir,
                      vocab_dir=merged_dir,
                      generate_basic_vocab=False,
                      load_vocab=True,
                      generate_tf_record=True,
                      max_document_length=max_document_length,
                      min_frequency=min_frequency,
                      max_frequency=max_frequency,
                      train_ratio=train_ratio,
                      valid_ratio=valid_ratio,
                      subsample_ratio=subsample_ratio,
                      padding=padding,
                      write_bow=write_bow
                      )
    vocab_v2i_dicts.append(dataset._categorical_vocab._mapping)
    vocab_i2v_lists.append(dataset._categorical_vocab._reverse_mapping)
    vocab_sizes.append(dataset._vocab_size)
    args_dicts.append(dataset._args)

  # tested
  # assert all(x == vocab_i2v_list[0] for x in vocab_i2v_list)
  # assert all(x == vocab_v2i_dict[0] for x in vocab_v2i_dict)
  # assert all(x == vocab_sizes[0] for x in vocab_sizes)

  with codecs.open(os.path.join(merged_dir, 'vocab_v2i.json'),
                   mode='w', encoding='utf-8') as file:
    json.dump(vocab_v2i_dicts[0], file, ensure_ascii=False, indent=4)
    file.close()

  vocab_i2v_dict = dict()
  for i in range(len(vocab_i2v_lists[0])):
    vocab_i2v_dict[i] = vocab_i2v_lists[0][i]
  with codecs.open(os.path.join(merged_dir, 'vocab_i2v.json'),
                   mode='w', encoding='utf-8') as file:
    json.dump(vocab_i2v_dict, file, ensure_ascii=False, indent=4)
    file.close()

  with open(os.path.join(merged_dir, "vocab_size.txt"), "w") as file:
    file.write(str(vocab_sizes[0]))
    file.close()

  return args_dicts


def get_types_and_counts(token_list):
  counts = {x: token_list.count(x) for x in token_list}
  return counts.keys(), counts.values()


def tokenizer(iterator):
  """Tokenizer generator.

  Tokenize each string with nltk's tweet_tokenizer, and add an 'EOS' at
  the end.

  Args:
    iterator: Input iterator with strings.

  Yields:
    array of tokens per each value in the input.
  """
  for value in iterator:
    yield value


def main():
  # test
  json_dirs = ["./vocab_test/1/", "./vocab_test/2/", "./vocab_test/3/"]
  tfrecord_dirs = ['./vocab_test/1/min_0_max_0/',
                   './vocab_test/2/min_0_max_0/',
                   './vocab_test/3/min_0_max_0/']
  merge_dict_write_tfrecord(json_dirs, tfrecord_dirs,
                            merged_dir="./vocab_test/merged/")


if __name__ == '__main__':
  main()

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

import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.contrib.learn.python.learn.preprocessing import \
    CategoricalVocabulary
from tqdm import tqdm

from tasks.code.data_prep import *
from tasks.code.text import VocabularyProcessor
from tasks.code.util import bag_of_words

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
                 valid_ratio=VALID_RATIO,
                 train_ratio=TRAIN_RATIO,
                 random_seed=RANDOM_SEED,
                 subsample_ratio=1,
                 padding=False,
                 write_bow=True
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
        :param padding: True if word_id needs padding to max_document_length
        """

        print("data in", json_dir)

        with gzip.open(os.path.join(json_dir, "data.json.gz"), "rt") as file:
            data = json.load(file)
            file.close()
        self._label_list = [int(item[label_field_name]) for item in data]
        self._num_classes = len(set(self._label_list))

        # if sys.version_info[0] < 3:
        #     self.text_list = df[text_field_names].astype(unicode).sum(
        #         axis=1).tolist()
        # else:
        #     self.text_list = df[text_field_names].astype(str).sum(
        #         axis=1).tolist()

        self.text_list = [' '.join([item[text_field_name]])
                          for text_field_name in text_field_names for
                          item in data]
        self.length_list = [len(text) for text in self.text_list]

        self.text_list = [tweet_tokenizer.tokenize(text) + ['EOS'] for text in
                          self.text_list]

        self.padding = padding

        # tokenize and reconstruct as string(which vocabulary processor
        # takes as input)

        # get index
        print("Generating train/valid/test splits...")
        index_path = os.path.join(json_dir, "index.json.gz")
        self.train_index, self.valid_index, self.test_index = self.split(
            index_path, train_ratio, valid_ratio, random_seed, subsample_ratio)

        # only compute from training data
        if max_document_length == -1:
            self.max_document_length = max(len(self.text_list[i]) for i in
                                           self.train_index)
            print("max document length (computed) =",
                  self.max_document_length)
        else:
            self.max_document_length = max_document_length
            print("max document length (given) =", self.max_document_length)
        print(self.max_document_length)

        self.vocab_dict = None
        self.categorical_vocab = None

        # generate and save the vocabulary which contains all the words
        if generate_basic_vocab:
            print("Generating the basic vocabulary.")
            self.build_save_basic_vocab(vocab_dir=json_dir)

        if generate_tf_record is False:
            print("No need to generate tf records. Done. ")
            return

        if load_vocab is False:
            print("No vocabulary given. Generate a new one.")
            self.categorical_vocab = self.build_vocab(
                min_frequency=min_frequency,
                max_frequency=max_frequency, vocab_dir=vocab_dir)
            self.save_vocab(tfrecord_dir)

        else:
            print("Public vocabulary given. Use that to build vocabulary "
                  "processor.")
            self.categorical_vocab = self.load_vocab(vocab_dir,
                                                     min_frequency=min_frequency,
                                                     max_frequency=max_frequency)
        # save mapping/reverse mapping to the disk
        # freq:            vocab_dir/vocab_freq.json
        # mapping:         vocab_dir/vocab_v2i.json
        # reverse mapping: vocab_dir/vocab_i2v.json(sorted according to freq)

        self.vocab_size = len(self.categorical_vocab._mapping)
        print("used vocab size =", self.vocab_size)

        # write to tf records
        try:
            os.stat(tfrecord_dir)
        except:
            os.mkdir(tfrecord_dir)
        self.train_path = os.path.join(tfrecord_dir, 'train.tf')
        self.valid_path = os.path.join(tfrecord_dir, 'valid.tf')
        self.test_path = os.path.join(tfrecord_dir, 'test.tf')

        print("Writing TFRecord files for the training data...")
        self.write_examples(self.train_path, self.train_index, write_bow)
        print("Writing TFRecord files for the validation data...")
        self.write_examples(self.valid_path, self.valid_index, write_bow)
        print("Writing TFRecord files for the test data...")
        self.write_examples(self.test_path, self.test_index, write_bow)

        # save dataset arguments
        self.args = {
            'num_classes': self._num_classes,
            'max_document_length': self.max_document_length,
            'vocab_size': self.vocab_size,
            'min_frequency': min_frequency,
            'max_frequency': max_frequency,
            'random_seed': random_seed,
            'train_path': os.path.abspath(self.train_path),
            'valid_path': os.path.abspath(self.valid_path),
            'test_path': os.path.abspath(self.test_path),
            'train_size': len(self.train_index),
            'valid_size': len(self.valid_index),
            'test_size': len(self.test_index)
        }
        print(self.args)
        args_path = os.path.join(tfrecord_dir, "args.json")
        with codecs.open(args_path, mode='w', encoding='utf-8') as file:
            json.dump(self.args, file, ensure_ascii=False, indent=4)
            file.close()

    def build_vocab(self, min_frequency, max_frequency, vocab_dir):
        """Builds vocabulary for this dataset only using tensorflow's
        VocabularyProcessor

        This vocabulary is only used for this dataset('s training data)
        """
        vocab_processor = VocabularyProcessor(
            max_document_length=self.max_document_length,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            tokenizer_fn=tokenizer)

        # build vocabulary only according to training data
        vocab_processor.fit([self.text_list[i] for i in self.train_index])

        if self.padding is True:
            self.word_id_list = list(
                vocab_processor.transform_pad(self.text_list))
        else:
            self.word_id_list = list(
                vocab_processor.transform(self.text_list))
        self.word_id_list = [list(i) for i in self.word_id_list]
        self.vocab_freq_dict = vocab_processor.vocabulary_._freq

        return vocab_processor.vocabulary_

    def save_vocab(self, vocab_dir):

        # save the built vocab to the disk for future use
        try:
            os.stat(vocab_dir)
        except:
            os.mkdir(vocab_dir)

        with codecs.open(os.path.join(vocab_dir, "vocab_freq.json"),
                         mode='w', encoding='utf-8')as file:
            json.dump(self.vocab_freq_dict, file, ensure_ascii=False, indent=4)
            file.close()

        with codecs.open(os.path.join(vocab_dir, "vocab_v2i.json"),
                         mode='w', encoding='utf-8')as file:
            json.dump(self.categorical_vocab._mapping, file,
                      ensure_ascii=False, indent=4)
            file.close()

        vocab_i2v_dict = dict()
        for i in range(len(self.categorical_vocab._reverse_mapping)):
            vocab_i2v_dict[i] = self.categorical_vocab._reverse_mapping[i]
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
            max_document_length=self.max_document_length,
            tokenizer_fn=tokenizer)

        # build vocabulary only according to training data
        vocab_processor.fit([self.text_list[i] for i in self.train_index])

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
                         mode="rt", encoding='utf-8') as file:
            self.vocab_freq_dict = json.load(file)
            file.close()
        categorical_vocab = CategoricalVocabulary()
        for word in self.vocab_freq_dict:
            categorical_vocab.add(word, count=self.vocab_freq_dict[word])
        categorical_vocab.trim(min_frequency=min_frequency,
                               max_frequency=max_frequency)
        categorical_vocab.freeze()

        vocab_processor = VocabularyProcessor(
            vocabulary=categorical_vocab,
            max_document_length=self.max_document_length,
            min_frequency=min_frequency,
            tokenizer_fn=tokenizer)

        if self.padding is True:
            self.word_id_list = list(
                vocab_processor.transform_pad(self.text_list))
        else:
            self.word_id_list = list(
                vocab_processor.transform(self.text_list))
        self.word_id_list = [list(i) for i in self.word_id_list]
        return vocab_processor.vocabulary_

    def write_examples(self, file_name, split_index, write_bow):
        # write to TFRecord data file
        tf.logging.info("Writing to: %s", file_name)
        with tf.python_io.TFRecordWriter(file_name) as writer:
            for index in tqdm(split_index):
                feature = {
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=[self._label_list[index]])),
                    'word_id': tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=self.word_id_list[index])),
                    'old_length': tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=[self.length_list[index]]))
                }

                if write_bow is True:
                    feature['bow'] = tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=bag_of_words(
                                self.word_id_list[index],
                                self.vocab_size).tolist()))

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature=feature))
                writer.write(example.SerializeToString())

    def split(self, index_path, train_ratio, valid_ratio, random_seed,
              subsample_ratio):
        if not Path(index_path).exists():
            # no split given
            print("no split given")
            train_index, valid_index, test_index \
                = self.random_split_train_valid_test(len(self.text_list),
                                                     train_ratio, valid_ratio,
                                                     random_seed)

        else:
            with gzip.open(index_path, mode='rt') as file:
                index = json.load(file)
                file.close()
            assert 'train' in index and 'test' in index
            train_index = index['train']
            test_index = index['test']
            if 'valid' in index:
                print("train/valid/test splits given")
                valid_index = index['valid']
            else:
                print("train/test splits given")
                train_index, valid_index = self.random_split_train_valid(
                    train_index, valid_ratio, random_seed)

        # no intersection
        assert (len(train_index) == len(set(train_index)))
        assert (len(valid_index) == len(set(valid_index)))
        assert (len(test_index) == len(set(test_index)))
        assert len([i for i in train_index if i in valid_index]) == 0
        assert len([i for i in train_index if i in test_index]) == 0
        assert len([i for i in valid_index if i in test_index]) == 0

        print("train : valid : test = %d : %d : %d" % (len(train_index),
                                                       len(valid_index),
                                                       len(test_index)))

        if subsample_ratio is not None and subsample_ratio < 1.0:
            train_index = self.subsample(train_index, random_seed,
                                         subsample_ratio)
            valid_index = self.subsample(valid_index, random_seed,
                                         subsample_ratio)
            test_index = self.subsample(test_index, random_seed,
                                        subsample_ratio)
            print("train : valid : test = %d : %d : %d" % (len(train_index),
                                                           len(valid_index),
                                                           len(test_index)))

        return train_index, valid_index, test_index

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


def merge_save_vocab_dicts(vocab_paths, save_path):
    """
    :param vocab_paths: list of vocabulary paths
    :return:
    """
    merged_vocab_dict = dict()
    for path in vocab_paths:
        vocab_dict = json.load(open(path, "rt"))
        merged_vocab_dict = combine_dicts(merged_vocab_dict, vocab_dict)

    print(merged_vocab_dict)

    with codecs.open(save_path, mode='w', encoding='utf-8') as file:
        json.dump(merged_vocab_dict, file, ensure_ascii=False, indent=4)
        file.close()


def combine_dicts(x, y):
    return {i: x.get(i, 0) + y.get(i, 0) for i in
            set(itertools.chain(x, y))}


def merge_dict_write_tfrecord(json_dirs, tfrecord_dirs, merged_dir,
                              max_document_length=-1, min_frequency=0,
                              max_frequency=-1, train_ratio=TRAIN_RATIO,
                              valid_ratio=VALID_RATIO, write_bow=True,
                              subsample_ratio=1):
    """
    1. generate and save vocab dictionary which contains all the words(
    cleaned) for each dataset
    2. merge the vocabulary
    3. generate and save TFRecord files for each dataset using the merged vocab
    :param json_dirs: list of dataset directories
    :param json_dirs: new directory to save all the data
    :return:
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
        max_document_lengths.append(dataset.max_document_length)

    # new data dir based all the datasets' names
    data_names = [os.path.basename(os.path.normpath(json_dir)) for json_dir
                  in json_dirs]
    data_names.sort()
    try:
        os.stat(merged_dir)
    except:
        os.mkdir(merged_dir)
    # new_data_dir = new_data_dir + '_'.join(data_names) + '/'
    # try:
    #     os.stat(new_data_dir)
    # except:
    #     os.mkdir(new_data_dir)

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
    args_lists = []

    # max_document_length is only useful when padding is True
    if max_document_length is None:
        max_document_length = max(max_document_lengths)
    for json_dir in json_dirs:
        tfrecord_dir = os.path.join(merged_dir, os.path.basename(
            os.path.normpath(json_dir)))
        dataset = Dataset(json_dir,
                          tfrecord_dir=tfrecord_dir,
                          vocab_dir=merged_dir,
                          max_document_length=max_document_length,
                          min_frequency=min_frequency,
                          max_frequency=max_frequency,
                          train_ratio=train_ratio,
                          valid_ratio=valid_ratio,
                          write_bow=write_bow,
                          subsample_ratio=subsample_ratio,
                          generate_basic_vocab=False,
                          load_vocab=True,
                          generate_tf_record=True
                          )
        vocab_v2i_dicts.append(dataset.categorical_vocab._mapping)
        vocab_i2v_lists.append(dataset.categorical_vocab._reverse_mapping)
        vocab_sizes.append(dataset.vocab_size)
        args_lists.append(dataset.args)

    # tested
    # assert all(x == vocab_i2v_list[0] for x in vocab_i2v_list)
    # assert all(x == vocab_v2i_dict[0] for x in vocab_v2i_dict)
    # assert all(x == vocab_sizes[0] for x in vocab_sizes)

    with open(os.path.join(merged_dir, 'vocab_v2i.json'),
              mode='w') as file:
        json.dump(vocab_v2i_dicts[0], file, ensure_ascii=False, indent=4)
        file.close()

    vocab_i2v_dict = dict()
    for i in range(len(vocab_i2v_lists[0])):
        vocab_i2v_dict[i] = vocab_i2v_lists[0][i]
    with open(os.path.join(merged_dir, 'vocab_i2v.json'),
              mode='w') as file:
        json.dump(vocab_i2v_dict, file, ensure_ascii=False, indent=4)
        file.close()

    with open(merged_dir + "vocab_size.txt", "w") as file:
        file.write(str(vocab_sizes[0]))
        file.close()

    return args_lists


# update_progress() : Displays or updates a console progress bar
# Accepts a float between 0 and 1. Any int will be converted to a float.
# A value under 0 represents a 'halt'.
# A value at 1 or bigger represents 100%

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

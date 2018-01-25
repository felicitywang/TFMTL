# Copyright 2017 Johns Hopkins University. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import itertools
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from data_prep import tweet_clean
from tensorflow.contrib.learn.python.learn.preprocessing import \
    CategoricalVocabulary
from text import VocabularyProcessor
from util import bag_of_words

flags = tf.flags
logging = tf.logging

FLAGS = flags.FLAGS


class Dataset():
    def __init__(self, data_dir, vocab_dir=None, tfrecord_dir=None,
                 max_document_length=None,
                 min_frequency=0, max_frequency=-1, encoding=None,
                 text_field_names=['text'],
                 label_field_name='label',
                 valid_ratio=0.1, train_ratio=0.8, random_seed=42):

        df = pd.read_json(data_dir + "data.json.gz")
        self.label_list = df[label_field_name].tolist()
        self.num_classes = len(set(self.label_list))
        self.text_list = df[text_field_names].astype(str).sum(axis=1).tolist()
        # for i in range(10):
        #     print(self.text_list[i])
        # # print(len(self.text_list))
        # for i in range(3):
        #     # print(self.text_list[i])

        self.encoding = encoding

        # tokenize and reconstruct as string(which vocabulary processor
        # takes as input)
        # TODO more on tokenizer
        self.text_list = [tweet_clean(i) for i in
                          self.text_list]
        # # print(len(self.text_list))
        # for i in range(3):
        #     # print(self.text_list[i])


        # get index
        index_path = os.path.join(data_dir, "index.json.gz")
        self.train_index, self.valid_index, self.test_index = self.split(
            index_path, train_ratio, valid_ratio, random_seed)

        if max_document_length is None:
            self.max_document_length = max(
                [len(x.split()) for x in self.text_list])
            print("max document length (computed) =",
                  self.max_document_length)
        else:
            self.max_document_length = max_document_length
            print("max document length (given) =", self.max_document_length)

        self.vocab_dict = None
        self.categorical_vocab = None

        self.save_vocab(vocab_dir=os.path.join(data_dir, "single/"))
        if vocab_dir is None:
            self.categorical_vocab = self.build_vocab(
                os.path.join(data_dir, "single/"),
                min_frequency=min_frequency,
                max_frequency=max_frequency)
        else:
            self.categorical_vocab = self.load_vocab(vocab_dir,
                                                     min_frequency=min_frequency,
                                                     max_frequency=max_frequency)
        print("-- freq:")
        print(self.categorical_vocab._freq)
        print("-- mapping:")
        print(self.categorical_vocab._mapping)
        self.vocab_size = len(self.categorical_vocab._mapping)

        # write to tf records
        if tfrecord_dir is None:
            tfrecord_dir = data_dir + 'single/'
        try:
            os.stat(tfrecord_dir)
        except:
            os.mkdir(tfrecord_dir)
        self.train_path = os.path.join(tfrecord_dir, 'train.tf')
        self.valid_path = os.path.join(tfrecord_dir, 'valid.tf')
        self.test_path = os.path.join(tfrecord_dir, 'test.tf')

        self.write_examples(self.train_path, self.train_index)
        self.write_examples(self.valid_path, self.valid_index)
        self.write_examples(self.test_path, self.test_index)

    def build_vocab(self, vocab_dir, min_frequency, max_frequency):
        """Builds vocabulary for this dataset only using tensorflow's
        VocabularyProcessor

        This vocabulary is only used for this dataset('s training data)
        Files saved to the disk should be:
        vocab_freq_dict_(min_freq).pickle
        vocab_i2v_list_(min_freq).pickle
        vocab_v2i_dict_(min_freq).pickle
        """
        vocab_processor = VocabularyProcessor(
            max_document_length=self.max_document_length,
            min_frequency=min_frequency,
            max_frequency=max_frequency)

        # build vocabulary only according to training data
        train_list = [self.text_list[i] for i in self.train_index]
        # print("train: ", train_list)
        vocab_processor.fit(train_list)

        self.word_id_list = list(
            vocab_processor.transform(self.text_list))
        self.word_id_list = [list(i) for i in self.word_id_list]
        self.vocab_freq_dict = vocab_processor.vocabulary_._freq

        # save
        try:
            os.stat(vocab_dir)
        except:
            os.mkdir(vocab_dir)
        with open(vocab_dir + "vocab_freq_dict_" + str(
                min_frequency) + ".pickle",
                  "wb") as file:
            pickle.dump(self.vocab_freq_dict, file)
            file.close()
        with open(vocab_dir + "vocab_v2i_dict_" + str(
                min_frequency) + ".pickle",
                  "wb") as file:
            pickle.dump(vocab_processor.vocabulary_._mapping, file)
            file.close()
        with open(vocab_dir + "vocab_i2v_list_" + str(
                min_frequency) + ".pickle",
                  "wb") as file:
            pickle.dump(vocab_processor.vocabulary_._reverse_mapping, file)
            file.close()

        return vocab_processor.vocabulary_

    def save_vocab(self, vocab_dir):
        """Bulid vocabulary with min_frequency=0 for this dataset'


        training data only and save to the directory
        minimum frequency is always 0 so that all the words of this dataset(
        's training data) are taken into account when merging with other
        vocabularies"""

        vocab_processor = VocabularyProcessor(
            max_document_length=self.max_document_length)

        # build vocabulary only according to training data
        train_list = [self.text_list[i] for i in self.train_index]
        # print("train: ", train_list)
        vocab_processor.fit(train_list)
        vocab_freq_dict = vocab_processor.vocabulary_._freq
        try:
            os.stat(vocab_dir)
        except:
            os.mkdir(vocab_dir)
        with open(vocab_dir + "vocab_freq_dict.pickle", "wb") as file:
            pickle.dump(vocab_freq_dict, file)
            file.close()

    def load_vocab(self, vocab_dir, min_frequency, max_frequency):
        with open(vocab_dir + "vocab_freq_dict.pickle", "rb") as file:
            self.vocab_freq_dict = pickle.load(file)
            file.close()
        categorical_vocab = CategoricalVocabulary()
        for word in self.vocab_freq_dict:
            categorical_vocab.add(word, count=self.vocab_freq_dict[word])
        categorical_vocab.trim(min_frequency=min_frequency,
                               max_frequency=max_frequency)
        categorical_vocab.freeze()

        # print("freq:")
        # print(categorical_vocab._freq)
        # print("mapping:")
        # print(categorical_vocab._mapping)

        vocab_processor = VocabularyProcessor(

            vocabulary=categorical_vocab,
            max_document_length=self.max_document_length,
            min_frequency=min_frequency)
        self.word_id_list = list(
            vocab_processor.transform(self.text_list))
        self.word_id_list = [list(i) for i in self.word_id_list]
        return vocab_processor.vocabulary_

    def write_examples(self, file_name, split_index):
        # write to TFRecord data file
        # tf.logging.info("Writing to: %s", file_name)
        with tf.python_io.TFRecordWriter(file_name) as writer:
            for index in split_index:
                feature = {
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=[self.label_list[index]])),
                    'word_id': tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=self.word_id_list[index])),
                    # 'bow': tf.train.Feature(
                    #     float_list=tf.train.FloatList(
                    #         value=bag_of_words(
                    #             self.word_id_list[index],
                    #             self.vocab_size).tolist()))
                }
                if self.encoding == 'bow':
                    feature['bow'] = tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=bag_of_words(
                                self.word_id_list[index],
                                self.vocab_size).tolist()))

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature=feature))
                writer.write(example.SerializeToString())

    def split(self, index_path, train_ratio, valid_ratio, random_seed):
        if not Path(index_path).exists():
            # no split given
            train_index, valid_index, test_index \
                = self.random_split_train_valid_test(len(self.text_list),
                                                     train_ratio, valid_ratio,
                                                     random_seed)
            print("train, valid, test index", train_index, valid_index,
                  test_index)
            # for i in train_index:
            #     print(self.text_list[i], self.label_list[i])
        else:
            index = json.load(gzip.open(index_path, mode='rt'))
            assert 'train' in index and 'test' in index
            train_index = index['train']
            test_index = index['test']
            if 'valid' in index:
                valid_index = index['valid']
            else:
                train_index, valid_index = self.random_split_train_valid(
                    train_index, valid_ratio, random_seed)

        # no intersection
        assert (len(train_index) == len(set(train_index)))
        assert (len(valid_index) == len(set(valid_index)))
        assert (len(test_index) == len(set(test_index)))
        assert len([i for i in train_index if i in valid_index]) == 0
        assert len([i for i in train_index if i in test_index]) == 0
        assert len([i for i in valid_index if i in test_index]) == 0

        # print("train : valid : test = %d : %d : %d" % (len(train_index),
        #                                                len(valid_index),
        #                                                len(test_index)))

        return train_index, valid_index, test_index

    def random_split_train_valid_test(self, length, train_ratio, valid_ratio,
                                      random_seed):
        index = np.array(list(range(length)))
        np.random.seed(random_seed)
        index = np.random.permutation(index)
        # # print(type(index))
        # # print(index)
        return np.split(index,
                        [int(train_ratio * len(index)),
                         int((train_ratio + valid_ratio) * len(index))])

    def random_split_train_valid(self, train_index, valid_ratio, random_seed):
        """Takes part of training data to validation data"""
        index = np.array(train_index)
        np.random.seed(random_seed)
        index = np.random.permutation(index)
        return np.split(index, [int(1.0 - valid_ratio * len(index))])


# add the frequencies of each in two vocabulary dictionary
def merge_vocab_dict(vocab_dir_1, vocab_dir_2):
    with open(vocab_dir_1 + "vocab_freq_dict.pickle", "rb") as file:
        vocab_freq_dict_1 = pickle.load(file)
        file.close()
    with open(vocab_dir_2 + "vocab_freq_dict.pickle", "rb") as file:
        vocab_freq_dict_2 = pickle.load(file)
        file.close()
    vocab_freq_dict = combine_dicts(vocab_freq_dict_1, vocab_freq_dict_2)
    return vocab_freq_dict


def combine_dicts(x, y):
    return {i: x.get(i, 0) + y.get(i, 0) for i in
            set(itertools.chain(x, y))}


def main():
    # combine dict
    # A = {'a': 1, 'b': 2, 'c': 3}
    # B = {'b': 3, 'c': 4, 'd': 5}
    # print(combine_dicts(A,B))

    # data_dir = "../datasets/other/AG_News/"
    # dataset = Dataset(data_dir=data_dir, text_field_names="title "
    #                                                       "description".split())

    # data_dir = "./cache/"
    # dataset = Dataset(data_dir=data_dir, text_field_names=['text'],
    #                   random_seed=random_seed)
    # data_dir = "../datasets/sentiment/SSTb/"
    # data_dir = "../datasets/sentiment/IMDB/"
    # dataset = Dataset(data_dir=data_dir, text_field_names=['text'])
    # print("index")
    # print(dataset.train_index)
    # print(dataset.valid_index)
    # print(dataset.test_index)

    data_dir_1 = "./vocab_test/1/"
    data_dir_2 = "./vocab_test/2/"

    random_seed = 3
    num = 3

    print("--- 1")
    dataset_1 = Dataset(data_dir=data_dir_1, text_field_names=['text'],
                        random_seed=random_seed)
    # print("--- 1 freq:", dataset_1.vocab_freq_dict)
    for i in range(num):
        print(dataset_1.text_list[i], dataset_1.word_id_list[i])

    print("--- 2")
    dataset_2 = Dataset(data_dir=data_dir_2, text_field_names=['text'],
                        random_seed=random_seed)
    # print("--- 1 freq:", dataset_2.vocab_freq_dict)
    for i in range(num):
        print(dataset_2.text_list[i], dataset_2.word_id_list[i])

    print("--- merged")
    vocab_freq_dict = merge_vocab_dict(vocab_dir_1=data_dir_1 + "single/",
                                       vocab_dir_2=data_dir_2 + "single/")

    print(vocab_freq_dict)
    vocab_dir = "./vocab_test/"
    with open(vocab_dir + "vocab_freq_dict.pickle", "wb") as file:
        pickle.dump(vocab_freq_dict, file)
        file.close()

    max_document_length = max(dataset_1.max_document_length,
                              dataset_2.max_document_length)
    print("--- 1 ")
    dataset_1 = Dataset(data_dir=data_dir_1, vocab_dir=vocab_dir,
                        text_field_names=['text'],
                        random_seed=random_seed,
                        max_document_length=max_document_length,
                        min_frequency=1,
                        max_frequency=4)
    print("--- 2")
    dataset_2 = Dataset(data_dir=data_dir_2, vocab_dir=vocab_dir,
                        text_field_names=['text'],
                        random_seed=random_seed,
                        max_document_length=max_document_length,
                        min_frequency=1,
                        max_frequency=4)


if __name__ == '__main__':
    main()

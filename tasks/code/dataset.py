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

import itertools
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from data_prep import tweet_clean
from tensorflow.contrib import learn
from util import bag_of_words


class Dataset():
    def __init__(self, data_dir, vocab_dir=None, tfrecord_dir=None,
                 max_document_length=None,
                 min_frequency=0, encoding=None, text_field_names=['text']):
        df = pd.read_json(data_dir + "data.json.gz")

        self.label_list = df.label.tolist()
        self.num_classes = len(set(self.label_list))
        self.text_list = df[text_field_names].astype(str).sum(axis=1).tolist()
        # for i in range(10):
        #     print(self.text_list[i])
        # # print(len(self.text_list))
        # for i in range(3):
        #     # print(self.text_list[i])

        self.encoding = encoding

        # tokenize and reconstruct as string
        # TODO more on tokenizer
        self.text_list = [tweet_clean(i) for i in
                          self.text_list]
        # # print(len(self.text_list))
        # for i in range(3):
        #     # print(self.text_list[i])

        if max_document_length is None:
            self.max_document_length = max(
                [len(x.split()) for x in self.text_list])
            print("max document length (computed) =",
                  self.max_document_length)
        else:
            self.max_document_length = max_document_length
            print("max document length (given) =", self.max_document_length)

        if vocab_dir is None:
            self.vocab_mapping = self.build_vocab(data_dir,
                                                  min_frequency)
        else:
            self.vocab_mapping = self.load_vocab(vocab_dir, min_frequency)
        self.vocab_size = len(self.vocab_mapping)

        # print(self.word_id_list)

        # TODO read index from disk
        train_index, valid_index, test_index \
            = random_train_validate_test_split(len(self.text_list))
        print("train size: ", len(train_index))
        print("valid size: ", len(valid_index))
        print("test  size: ", len(test_index))
        # print(train_index, validate_index, test_index)
        # for i in train_index:
        #     print(self.text_list[i], self.label_list[i])

        # write to tf records

        if tfrecord_dir is None:
            tfrecord_dir = data_dir + 'tfrecords/'
        try:
            os.stat(tfrecord_dir)
        except:
            os.mkdir(tfrecord_dir)
        self.train_path = os.path.join(tfrecord_dir, 'train.tf')
        self.valid_path = os.path.join(tfrecord_dir, 'valid.tf')
        self.test_path = os.path.join(tfrecord_dir, 'test.tf')

        self.write_examples(self.train_path, train_index)
        self.write_examples(self.valid_path, valid_index)
        self.write_examples(self.test_path, test_index)

    def build_vocab(self, vocab_dir, min_frequency):
        # build vocab of only this dataset and save to disk
        vocab_processor = learn.preprocessing.VocabularyProcessor(
            max_document_length=self.max_document_length,
            min_frequency=min_frequency)

        self.word_id_list = list(
            vocab_processor.fit_transform(self.text_list))
        self.word_id_list = [list(i) for i in self.word_id_list]
        # print(self.word_ids)
        # self.word_ids = [i.tolist() for i in self.word_ids]
        # save categorical vocabulary to disk
        # python dict {word:freq}
        vocab_dict = vocab_processor.vocabulary_._freq
        # print("freq:")
        # print(vocab_dict)
        # print("mapping:")
        # print(vocab_processor.vocabulary_._mapping)
        # print("reverse mapping:")
        # print(vocab_processor.vocabulary_._reverse_mapping)

        with open(vocab_dir + "vocab_dict.pickle", "wb") as file:
            pickle.dump(vocab_dict, file)
            file.close()
        return vocab_processor.vocabulary_._mapping

    def load_vocab(self, vocab_dir, min_frequency):
        with open(vocab_dir + "vocab_dict.pickle", "rb") as file:
            vocab_dict = pickle.load(file)
            file.close()
        categorical_vocab = learn.preprocessing.CategoricalVocabulary()
        for word in vocab_dict:
            categorical_vocab.add(word, count=vocab_dict[word])
        if min_frequency is not None:
            categorical_vocab.trim(min_frequency)
        categorical_vocab.freeze()
        # print("dict")
        # print(vocab_dict)
        # print("freq:")
        # print(categorical_vocab._freq)
        # print("mapping:")
        # print(categorical_vocab._mapping)
        vocab_processor = learn.preprocessing.VocabularyProcessor(
            vocabulary=categorical_vocab,
            max_document_length=self.max_document_length,
            min_frequency=min_frequency)
        self.word_id_list = list(
            vocab_processor.fit_transform(self.text_list))
        self.word_id_list = [list(i) for i in self.word_id_list]
        return vocab_processor.vocabulary_._mapping

    def write_examples(self, file_name, split_index):
        # write to TFRecord data file
        tf.logging.info("Writing to: %s", file_name)
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

                # TODO add fraction variables


def random_train_validate_test_split(length):
    index = np.array(list(range(length)))
    index = np.random.permutation(index)
    # # print(type(index))
    # # print(index)
    return np.split(index,
                    [int(.6 * len(index)),
                     int(.8 * len(index))])


# add the frequencies of each in two vocabulary dictionary
def merge_vocab_dict(vocab_dir_1, vocab_dir_2):
    with open(vocab_dir_1 + "vocab_dict.pickle", "rb") as file:
        vocab_dict_1 = pickle.load(file)
        file.close()
    with open(vocab_dir_2 + "vocab_dict.pickle", "rb") as file:
        vocab_dict_2 = pickle.load(file)
        file.close()
    vocab_dict = combine_dicts(vocab_dict_1, vocab_dict_2)
    return vocab_dict


def combine_dicts(x, y):
    z = {i: x.get(i, 0) + y.get(i, 0) for i in set(itertools.chain(x, y))}
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
    data_dir = "./cache/"
    dataset = Dataset(data_dir=data_dir, text_field_names=['text'])


if __name__ == '__main__':
    main()

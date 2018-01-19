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
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from data_prep import tweet_clean
from tensorflow.contrib import learn


class Dataset():
    def __init__(self, data_dir, vocab_dir=None, max_document_length=None,
                 min_frequency=0):
        df = pd.read_json(data_dir + "data.json")

        self.label_list = df.label.tolist()

        self.text_list = df.text.tolist()
        # # print(len(self.text_list))
        # for i in range(3):
        #     # print(self.text_list[i])

        # tokenize and reconstruct as string
        self.text_list = [tweet_clean(i) for i in
                          self.text_list]
        # # print(len(self.text_list))
        # for i in range(3):
        #     # print(self.text_list[i])


        self.word_ids = None

        if max_document_length is None:
            max_document_length = max(
                [len(x.split()) for x in self.text_list])
            print("max document length (computed)= ", max_document_length)
        else:
            print("max document length (given)= ", max_document_length)

        if vocab_dir is None:
            self.build_vocab(data_dir,
                             max_document_length, min_frequency)
        else:
            self.load_vocab(vocab_dir, max_document_length, min_frequency)

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
        self.write_examples(data_dir + "train.tfrecord", train_index)
        self.write_examples(data_dir + "valid.tfrecord", valid_index)
        self.write_examples(data_dir + "test.tfrecord", test_index)

    def build_vocab(self, vocab_dir, max_document_length, min_frequency):
        # build vocab of only this dataset and save to disk
        vocab_processor = learn.preprocessing.VocabularyProcessor(
            max_document_length=max_document_length)
        self.word_ids = list(vocab_processor.fit_transform(self.text_list))
        self.word_ids = [i.tolist() for i in self.word_ids]
        # save categorical vocabulary to disk
        # python dict {word:freq}
        vocab_dict = vocab_processor.vocabulary_._freq
        # print("freq:")
        # print(vocab_dict)
        # print("mapping:")
        # print(vocab_processor.vocabulary_._mapping)
        with open(vocab_dir + "vocab_dict.pickle", "wb") as file:
            pickle.dump(vocab_dict, file)
            file.close()

    def load_vocab(self, vocab_dir, max_document_length, min_frequency):
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
            max_document_length=max_document_length,
            min_frequency=min_frequency)
        self.word_ids = list(vocab_processor.fit_transform(self.text_list))
        self.word_ids = [i.tolist() for i in self.word_ids]

    def write_examples(self, file_name, split_index):
        # write to TFRecord data file
        tf.logging.info("Writing to: %s", file_name)
        with tf.python_io.TFRecordWriter(file_name) as writer:
            for index in split_index:
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'label': tf.train.Feature(
                                int64_list=tf.train.Int64List(
                                    value=[self.label_list[index]])),
                            'text': tf.train.Feature(
                                int64_list=tf.train.Int64List(
                                    value=self.word_ids[index]))
                        }))
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
    print(x)
    print(y)
    z = {i: x.get(i, 0) + y.get(i, 0) for i in set(itertools.chain(x, y))}
    print(z)
    return {i: x.get(i, 0) + y.get(i, 0) for i in set(itertools.chain(x, y))}


def main():
    # combine dict
    # A = {'a': 1, 'b': 2, 'c': 3}
    # B = {'b': 3, 'c': 4, 'd': 5}
    # print(combine_dicts(A,B))
    data_dir = "./cache/"
    dataset = Dataset(data_dir=data_dir)

    # build = Dataset(data_dir=data_dir)
    # load = Dataset(data_dir=data_dir, vocab_dir=data_dir)

    # print(build.word_ids)
    # print(load.word_ids)

    # # print(build.word_id_list[0])
    # # print(load.word_id_list[0])
    # assert np.array_equal(build.word_id_list, load.word_id_list)
    FEATURES = {
        'text': tf.FixedLenFeature([], dtype=tf.int64),
        'label': tf.FixedLenFeature([], dtype=tf.int64)
    }

    # TODO test data
    # with tf.Graph().as_default():
    #     train_path = "./cache/train.tfrecord"
    #     valid_path = "./cache/valid.tfrecord"
    #     test_path = "./cache/test.tfrecord"
    #
    #     train = InputDataset(train_path, FEATURES, 32)
    #     train_batch = train.batch
    #     train_init = train.init_op
    #
    #     text = train_batch['text']
    #     label = train_batch['label']
    #


if __name__ == '__main__':
    main()

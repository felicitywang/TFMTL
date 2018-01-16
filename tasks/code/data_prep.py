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
"""Data reprocessing

transform text data into n-gram models
split data into train/dev/test sets
"""

import numpy as np
import pandas as pd
import scipy
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

# stemmer and tokenizer in NLTK
from sklearn.preprocessing import LabelEncoder

stemmer = PorterStemmer()


def stem_tokens(tokens, porter_stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(porter_stemmer.stem(item))
    return stemmed


tweet_tokenizer = TweetTokenizer(strip_handles=True,
                                 preserve_case=False,
                                 reduce_len=True)  # e.g. waaaayyyyyy -> waayyy


def my_tokenizer(text):
    # remove punctuations other than ?!.
    # remove urls
    # text = re.sub(
    #     r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,'
    #     r'.;]+[-A-Za-z0-9+&@#/%=~_|][\r\n]*',
    #     ' ', text, flags=re.MULTILINE)
    tokens = tweet_tokenizer.tokenize(text)

    # stems = stem_tokens(tokens, stemmer)
    # return stems

    return tokens


# transform data['text'](string) to ngram model using
# sklearn.feature_extraction.text.TfidfVectorizer

def transform_data(data_dir, index_data_name,
                   mtx_data_name, min_ngram=1, max_ngram=3, min_df=50):
    data = pd.read_json(path_or_buf=data_dir + index_data_name)

    # texts = my_tokenizer(data['text']).values.tolist()
    # tfidf using keras
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(texts)
    # encoded_texts = tokenizer.texts_to_matrix(texts, mode='tfidf')

    # tfidf using sklearn
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=my_tokenizer,
        ngram_range=(min_ngram, max_ngram),
        min_df=min_df
    )
    tfidf_data = tfidf_vectorizer.fit_transform(data['text'])
    scipy.io.mmwrite(data_dir + mtx_data_name, tfidf_data)
    return tfidf_data


# randomly split data into train, dev, test = 3:1:1
# using sklearn.model_selection.KFold
def random_split_data(data_dir, index_data_name, mtx_data_name):
    # load data
    x = pd.read_json(path_or_buf=data_dir + index_data_name)
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(x.label)
    # y_all = x.label
    # output_size = len(list(y_all.unique()))

    # transform
    # transform_data(data_dir, index_data_name, mtx_data_name)
    x_data = scipy.io.mmread(data_dir + mtx_data_name)
    x_data = x_data.tocsr()

    # split data into (train, dev), (test) sets
    kfold_test = KFold(n_splits=5, shuffle=True)
    index, test_index = next(kfold_test.split(x_data))

    x, y = x_data[index], y_all[index]
    x_test, y_test = x_data[test_index], y_all[test_index]

    # split left data into train and dev sets
    kfold_dev = KFold(n_splits=4, shuffle=True)

    # run once only
    train_index, dev_index = next(kfold_dev.split(y))

    x_train, y_train = x[train_index], y[train_index]
    x_dev, y_dev = x[dev_index], y[dev_index]

    # print('X shape:', X.shape)
    # print('y shape:', y.shape)
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_dev  shape:', x_dev.shape)
    print('y_dev  shape:', y_dev.shape)
    print('x_test  shape:', x_test.shape)
    print('y_test  shape:', y_test.shape)

    output_size = len(label_encoder.classes_)
    return x_train, y_train, x_dev, y_dev, x_test, y_test, output_size


# split data according to given train/dev/test indices
def split_data(data_dir, index_data_name, mtx_data_name, index_dict):
    # load data
    x = pd.read_json(path_or_buf=data_dir + index_data_name)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(x.label)
    # y = x.label
    # output_size = len(list(y.unique()))

    x_data = scipy.io.mmread(data_dir + mtx_data_name)
    x_data = x_data.tocsr()

    train_index = np.array(index_dict['train_index'])
    dev_index = np.array(index_dict['dev_index'])
    test_index = np.array(index_dict['test_index'])

    x_train, y_train = x_data[train_index], y[train_index]
    x_dev, y_dev = x_data[dev_index], y[dev_index]
    x_test, y_test = x_data[test_index], y[test_index]

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_dev  shape:', x_dev.shape)
    print('y_dev  shape:', y_dev.shape)
    print('x_test  shape:', x_test.shape)
    print('y_test  shape:', y_test.shape)

    output_size = len(label_encoder.classes_)
    return x_train, y_train, x_dev, y_dev, x_test, y_test, output_size

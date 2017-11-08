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
"""Extract features and labels from original data

Features:

    Content+Punctuation()
    Structure:  depth(raw count + normalized),
                number of sentences/words/characters of both body and title,
                for both current and parent
    Author: whether author of initial post / same as the parent commenter
    Thread: total number of comments in the discussion,
            whether self_post / link_post
            average length of all the branches/threads of discussion in the
            discussion tree
    Community: subreddit
"""

import pickle
import random
# sys.path.append(os.path.abspath('.'))
import time

import pandas as pd
import scipy
from my_tokenizer import tokenizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# from transformers import *

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder()
one_hot_encoder_dense = OneHotEncoder(sparse=False)


# class TextExtractor(BaseEstimator, TransformerMixin):
#     def __init__(self, column):
#         self.column = column
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         return np.asarray(X[self.column]).astype(str)


class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[:, self.column]


class SubredditEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = label_encoder.fit_transform(X).reshape(-1, 1)
        return one_hot_encoder.fit_transform(data)


num_feature_list = ['num_sent', 'num_word', 'num_char', 'post_depth',
                    'post_depth_normalized',
                    'parent_num_sent', 'parent_num_word', 'parent_num_char',
                    'parent_post_depth', 'parent_post_depth_normalized',
                    'thread_comment_num', 'thread_branch_num',
                    'thread_branch_len']

bool_feature_list = ['is_first_author', 'is_initial_author',
                     'is_parent_author', 'thread_is_self_post']

tfidf_vectorizer = TfidfVectorizer(
    tokenizer=tokenizer,
    # stop_words='english',
    # vocabulary=dictionary,
    ngram_range=(1, 3),
    min_df=50
    # min_df=20
)

# TODO separate body and title (?)cv
features = FeatureUnion(
    transformer_list=[
        ('text_features', Pipeline([
            ('text_extractor', ColumnExtractor('text')),
            ('tfidf_vectorizer', tfidf_vectorizer)
        ])),
        ('parent_text_features', Pipeline([
            ('text_extractor', ColumnExtractor('parent_text')),
            ('tfidf_vectorizer', tfidf_vectorizer)
        ])),
        # ('body_features', Pipeline([
        #     ('text_extractor', TextExtractor('body')),
        #     ('tfidf_vectorizer', tfidf_vectorizer)
        # ])),
        # ('title_features', Pipeline([
        #     ('text_extractor', TextExtractor('title')),
        #     ('tfidf_vectorizer', tfidf_vectorizer)
        # ])),
        ('num_features', Pipeline([
            ('column_extractor', ColumnExtractor(num_feature_list)),
            ('scaler', StandardScaler())
        ])),
        ('bool_features', Pipeline([
            ('column_extractor', ColumnExtractor(bool_feature_list)),
            ('imputer', Imputer())
        ])),
        ('subreddit_features', Pipeline([
            ('column_extractor', ColumnExtractor('subreddit')),
            ('subreddit_encoder', SubredditEncoder())
        ])),
        # ('parent_text_features', Pipeline([
        #     ('text_extractor', ColumnExtractor('parent_text')),
        #     ('tfidf_vectorizer', tfidf_vectorizer)
        # ])),

    ],
    transformer_weights={
        'text_features': 0.8,
        'parent_text_features': 0.6,
        # 'body_features': 0.8,
        # 'title_features': 0.4,
        'num_features': 0.4,
        'bool_features': 0.2,
        'subreddit_features': 0.1
    },
    n_jobs=-1
)

# group 10-fold cross validation based on thread_id
# to make sure discussions in the thread are trained together
group_k_fold_10 = GroupKFold(n_splits=10)

# 10-fold with shuffle and random state
k_fold_10 = KFold(n_splits=10, shuffle=True,
                  random_state=random.randint(0, 9999))

# models
logreg = LogisticRegression(
    penalty='l2',
    solver='liblinear',
    C=3.0
    # verbose=True
)

mlp = MLPClassifier(verbose=True, hidden_layer_sizes=(200, 200),
                    solver='adam', alpha=0.001, activation="logistic",
                    learning_rate='invscaling', learning_rate_init=0.01)


def prepare_data_save(data_dir):
    post_list = pickle.load(open(data_dir + "post_list.pickle", "rb"))
    thread_list = pickle.load(open(data_dir + "thread_list.pickle", "rb"))

    post_df = pd.DataFrame(post_list)
    thread_df = pd.DataFrame(thread_list)

    post_df = post_df.set_index('id')
    thread_df = thread_df.set_index('id')

    # post_df['thread_is_self_post'] = None
    # post_df['thread_avg_num_sent'] = None
    # post_df['thread_avg_num_word'] = None
    # post_df['thread_avg_num_char'] = None
    # post_df['thread_avg_num_post_depth'] = None


    # Thread features
    post_df['thread_comment_num'] = None
    post_df['thread_branch_num'] = None
    post_df['thread_branch_len'] = None

    # Structure features
    post_df['parent_num_sent'] = None
    post_df['parent_num_word'] = None
    post_df['parent_num_char'] = None
    post_df['parent_post_depth'] = None
    post_df['parent_post_depth_normalized'] = None

    for index, row in post_df.iterrows():
        # Thread features
        post_df.set_value(index, 'thread_is_self_post',
                          thread_df.loc[row.thread_id].is_self_post)
        post_df.set_value(index, 'thread_comment_num',
                          thread_df.loc[row.thread_id].num_comments)
        post_df.set_value(index, 'thread_branch_num',
                          thread_df.loc[row.thread_id].num_branches)
        post_df.set_value(index, 'thread_branch_len',
                          thread_df.loc[row.thread_id].avg_len_branches)

        # Structure features
        # set to 0 if comment doesn't have a comment

        parent_id = row.in_reply_to
        post_df.set_value(
            index, 'parent_num_sent',
            0 if parent_id != parent_id or parent_id not in post_df.index else
            post_df.loc[parent_id].num_sent)
        post_df.set_value(
            index, 'parent_num_word',
            0 if parent_id != parent_id or parent_id not in post_df.index else
            post_df.loc[parent_id].num_word)
        post_df.set_value(
            index, 'parent_num_char',
            0 if parent_id != parent_id or parent_id not in post_df.index else
            post_df.loc[parent_id].num_char)
        post_df.set_value(
            index, 'parent_post_depth',
            0 if parent_id != parent_id or parent_id not in post_df.index else
            post_df.loc[parent_id].post_depth)
        post_df.set_value(
            index, 'parent_post_depth_normalized',
            0 if parent_id != parent_id or parent_id not in post_df.index else
            post_df.loc[parent_id].post_depth_normalized)

        # post_df.set_value(index, 'thread_avg_num_sent',
        #                   thread_df.loc[row.thread_id].avg_num_sent)
        # post_df.set_value(index, 'thread_avg_num_word',
        #                   thread_df.loc[row.thread_id].avg_num_word)
        # post_df.set_value(index, 'thread_avg_num_char',
        #                   thread_df.loc[row.thread_id].avg_num_char)

        # if row.is_first_post:
        #     title = thread_df.loc[row.thread_id].title
        #     post_df.set_value(index, 'new_title', title)
        #     if row.body == row.body:
        #         post_df.set_value(index, 'new_text',
        #                           str(title) + " " + str(row.body))
        # elif row.body == row.body:
        #         post_df.set_value(index, 'new_text', row.body)

        # test of dataframe
        # post_list = post_list[:1000]
    post_df_path = data_dir + "post_df.json"
    thread_df_path = data_dir + "thread_df.json"
    post_df.to_json(path_or_buf=post_df_path, orient='index')
    thread_df.to_json(path_or_buf=thread_df_path, orient='index')

    # add text of the link in reply to
    # none if is initial post
    for index, row in post_df.iterrows():
        # Thread features
        parent_id = row.in_reply_to
        post_df.set_value(
            index, 'parent_text',
            "" if parent_id != parent_id or parent_id not in
                                            post_df.index
            else
            post_df.loc[parent_id].text)
    post_df_path = data_dir + "post_df_parent_text_none.json"
    post_df.to_json(path_or_buf=post_df_path, orient='index')

    # add text of the link in reply to
    # self text if initial post
    for index, row in post_df.iterrows():
        # Thread features
        parent_id = row.in_reply_to
        post_df.set_value(
            index, 'parent_text',
            row.text if parent_id != parent_id or parent_id not in
                                                  post_df.index
            else
            post_df.loc[parent_id].text)
    post_df_path = data_dir + "post_df_parent_text.json"
    post_df.to_json(path_or_buf=post_df_path, orient='index')


def prepare_data_load(data_file_name="../data/post_df_parent_text.json"):
    return pd.read_json(path_or_buf=data_file_name, orient='index')


def load_data(data_file_name):
    """return DataFrame X and y"""
    # prepare_data_save()
    X = prepare_data_load(data_file_name)
    # X = X[X.text.notnull()]
    X = X[X.majority_type.notnull()]
    X = X[X.majority_type != 'other']
    X.label = label_encoder.fit_transform(X.majority_type)
    y = X.label
    print("Labels are", label_encoder.classes_)
    return X, y


def transform_data(X, data_file_name):
    print("Transforming data ...")
    ticks = time.clock()
    X_data = features.fit_transform(X)
    print("time used to transform the data is %.2f s" %
          (time.clock() - ticks))
    scipy.io.mmwrite(data_file_name, X_data)


# split data for tensorflow
def split_data(base_dir,index_data_name, mtx_data_name):
    # load data
    data_dir = base_dir + "data/"

    print("Loading data ...")
    _, y_all = load_data(data_dir + index_data_name)

    X_data = scipy.io.mmread(data_dir + mtx_data_name)
    X_data = X_data.tocsr()

    # split data into (train, dev), (test) sets
    kfold_test = KFold(n_splits=5, shuffle=True)
    index, test_index = next(kfold_test.split(X_data, y_all))

    X, y = X_data[index], y_all[index]
    X_test, y_test = X_data[test_index], y_all[test_index]

    # print('X shape:', X.shape)
    # print('y shape:', y.shape)
    # print('X_test  shape:', X_test.shape)
    # print('y_test  shape:', y_test.shape)

    # split left data into train and dev sets
    kfold_dev = KFold(n_splits=5, shuffle=True)

    # run once only
    train_index, dev_index = next(kfold_dev.split(X, y))

    X_train, y_train = X[train_index], y[train_index]
    X_dev, y_dev = X[dev_index], y[dev_index]

    # print('X_train shape:', X_train.shape)
    # print('y_train shape:', y_train.shape)
    # print('X_dev  shape:', X_dev.shape)
    # print('y_dev  shape:', y_dev.shape)

    return X_train, y_train, X_dev, y_dev, X_test, y_test

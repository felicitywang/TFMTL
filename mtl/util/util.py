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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading
from datetime import datetime
from datetime import timedelta

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def create_timestamp_bins(times):
  """Given a list of timestamps, create a dictionary mapping days in
  the range to a sequence of monotonically increasing integers.

  Returns: a tuple of (dict, int) where
    dict: mapping from timestamp to integer
    int: number of bins
  """
  max_time = datetime.fromtimestamp(max(times))
  min_time = datetime.fromtimestamp(min(times))
  date_to_bin = {}
  curr_time = min_time
  curr_bin = 0
  while curr_time < max_time:
    m = curr_time.month
    d = curr_time.day
    date_to_bin[(m, d)] = curr_bin
    curr_time += timedelta(days=1)
    curr_bin += 1
  m = curr_time.month
  d = curr_time.day
  date_to_bin[(m, d)] = curr_bin
  return date_to_bin


def hours_and_minutes(elapsed_sec):
  sec = timedelta(seconds=elapsed_sec).seconds
  return sec // 3600, sec // 60 % 60


def get_dropout_mask(keep_prob, shape):
  keep_prob = tf.convert_to_tensor(keep_prob)
  random_tensor = keep_prob + tf.random_uniform(shape)
  binary_tensor = tf.floor(random_tensor)
  dropout_mask = tf.inv(keep_prob) * binary_tensor
  return dropout_mask


class threadsafe_iter:
  """Takes an iterator/generator and makes it thread-safe by serializing
    call to the `next` method of given iterator/generator.

  """

  def __init__(self, it):
    self.it = it
    self.lock = threading.Lock()

  def __iter__(self):
    return self

  def __next__(self):
    with self.lock:
      return next(self.it)


def threadsafe_generator(f):
  def g(*a, **kw):
    return threadsafe_iter(f(*a, **kw))

  return g


def bag_of_words(words, vocab_size, freq=False, norm=True, dtype=np.float32):
  """This assumes words are integers."""
  if type(words) != list:
    raise ValueError("words should be list")

  if len(words) < 1:
    raise ValueError("empty word sequence")

  # if type(words[0]) != int:
  #     raise ValueError("must provide integer sequences")

  X = np.zeros(vocab_size, dtype=dtype)
  if freq:
    for word in words:
      X[word] += 1
  else:
    types = set(words)
    for word in types:
      X[word] = 1

  if norm:
    denom = np.linalg.norm(X)
    denom += np.finfo(X.dtype).eps
    X = X / denom

  return X


# TF-IDF
# https://stevenloria.com/tf-idf/
# https://gist.github.com/anabranch/48c5c0124ba4e162b2e3

def _jaccard_similarity(query, document):
  intersection = set(query).intersection(set(document))
  union = set(query).union(set(document))
  return len(intersection) / len(union)


def _term_frequency(term, tokenized_document):
  return tokenized_document.count(term)


def _sublinear_term_frequency(term, tokenized_document):
  count = tokenized_document.count(term)
  if count == 0:
    return 0
  return 1 + np.log(count)


def _augmented_term_frequency(term, tokenized_document):
  max_count = max([_term_frequency(t, tokenized_document)
                   for t in tokenized_document])
  return 0.5 + (
    (0.5 * _term_frequency(term, tokenized_document)) / max_count)


def _inverse_document_frequencies(tokenized_documents, vocab=None):
  idf_values = {}
  if not vocab:
    vocab = set([item for sublist in tokenized_documents
                 for item in sublist])
  # print("vocab", vocab)
  print("Generating idf values...")
  for token in tqdm(vocab):
    contains_token = map(lambda doc: token in doc, tokenized_documents)
    # print(sum(contains_token))
    sum_contains_token = sum(contains_token)
    if sum_contains_token == 0:
      sum_contains_token = 1
    idf_values[token] = 1 + np.log(
      len(tokenized_documents) / sum_contains_token)
  return idf_values


def tfidf(tokenized_documents, vocab=None):
  # tokenized_documents = [tokenize(d) for d in documents]
  idf = _inverse_document_frequencies(tokenized_documents, vocab)
  tfidf_documents = []
  print("Generating tfidf features...")
  for document in tqdm(tokenized_documents):
    doc_tfidf = []
    for term in idf.keys():
      tf = _sublinear_term_frequency(term, document)
      doc_tfidf.append(tf * idf[term])
    tfidf_documents.append(doc_tfidf)
  return tfidf_documents


def _cosine_similarity(vector1, vector2):
  dot_product = sum(p * q for p, q in zip(vector1, vector2))
  magnitude = (np.sqrt(sum([val ** 2 for val in vector1])) *
               np.sqrt(sum([val ** 2 for val in vector2])))
  if not magnitude:
    return 0
  return dot_product / magnitude


def make_dir(dir):
  try:
    os.stat(dir)
  except OSError:
    os.makedirs(dir)


if __name__ == "__main__":
  """Test bag of words"""
  # words = [1, 2, 3, 4, 4, 5]
  # vocab_size = 10
  # X1 = bag_of_words(words, vocab_size, norm=True)
  # print(X1)
  # X2 = bag_of_words(words, vocab_size, norm=False)
  # print(X2)

  """Test Tf-idf"""
  all_documents = ['a b b c c c EOS', 'd e f g EOS', 'a b c d e f g h EOS']


  def tokenize(doc):
    return doc.lower().split(" ")


  # in Scikit-Learn
  from sklearn.feature_extraction.text import TfidfVectorizer

  sklearn_tfidf = TfidfVectorizer(norm='l2', min_df=0, use_idf=True,
                                  smooth_idf=False, sublinear_tf=True,
                                  tokenizer=tokenize)
  sklearn_representation = sklearn_tfidf.fit_transform(all_documents)

  tokenized_documents = [tokenize(document) for document in all_documents]
  tfidf_representation = tfidf(tokenized_documents)

  our_tfidf_comparisons = []
  for count_0, doc_0 in enumerate(tfidf_representation):
    for count_1, doc_1 in enumerate(tfidf_representation):
      our_tfidf_comparisons.append((_cosine_similarity(doc_0, doc_1),
                                    count_0, count_1))

  skl_tfidf_comparisons = []
  for count_0, doc_0 in enumerate(sklearn_representation.toarray()):
    for count_1, doc_1 in enumerate(sklearn_representation.toarray()):
      skl_tfidf_comparisons.append((_cosine_similarity(doc_0, doc_1),
                                    count_0, count_1))

  for x in zip(sorted(our_tfidf_comparisons, reverse=True),
               sorted(skl_tfidf_comparisons, reverse=True)):
    print(x)

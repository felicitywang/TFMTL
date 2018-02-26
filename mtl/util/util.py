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

import threading
from datetime import datetime
from datetime import timedelta

import numpy as np
import tensorflow as tf


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


if __name__ == "__main__":
    words = [1, 2, 3, 4, 4, 5]
    vocab_size = 10
    X1 = bag_of_words(words, vocab_size, norm=True)
    print(X1)
    X2 = bag_of_words(words, vocab_size, norm=False)
    print(X2)

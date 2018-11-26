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
"""Transform original LMRD files into json format"""

# -*- coding: utf-8 -*-

import gzip
import json
import os
import sys
from pprint import pprint

from tqdm import tqdm


def main():
  dir = sys.argv[1]
  train_dir = os.path.join(dir, '20news-bydate-train')
  test_dir = os.path.join(dir, '20news-bydate-test')

  # get labels
  topics = []
  for topic in os.listdir(train_dir):
    topics.append(topic)
  topics.sort()
  label_dict = dict()
  for label, topic in enumerate(topics):
    label_dict[topic] = label
  pprint(label_dict)

  train_list = []
  test_list = []

  index = 0

  # train
  for topic in tqdm(os.listdir(train_dir)):
    for filename in os.listdir(os.path.join(train_dir, topic)):
      with open(os.path.join(train_dir, topic, filename), encoding='latin-1') \
        as file:
        text = file.read()
        train_list.append({
          'text': text,
          'topic': topic,
          'label': label_dict[topic],
          'index': index,
          'id': filename
        })
      index += 1

  # test
  for topic in tqdm(os.listdir(test_dir)):
    for filename in os.listdir(os.path.join(test_dir, topic)):
      with open(os.path.join(test_dir, topic, filename), encoding='latin-1') \
        as file:
        text = file.read()
        test_list.append({
          'text': text,
          'topic': topic,
          'label': label_dict[topic],
          'index': index,
          'id': filename
        })
      index += 1

  train_index = list(range(len(train_list)))
  test_index = list(range(len(train_list), len(train_list) + len(test_list)))

  index_dict = {
    'train': train_index,
    'test': test_index
  }
  assert len(set(index_dict['train']).intersection(index_dict['test'])) == 0

  # save
  with gzip.open('index.json.gz', mode='wt') as filename:
    json.dump(index_dict, filename, ensure_ascii=False)

  train_list.extend(test_list)
  with gzip.open('data.json.gz', mode='wt') as filename:
    json.dump(train_list, filename, ensure_ascii=False)

if __name__ == '__main__':
  main()

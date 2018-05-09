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

"""Transform original SSTb file into json format using pytreebank parser"""

# -*- coding: utf-8 -*-

import gzip
import json
# path that saves
import sys

import pytreebank

path = sys.argv[1]
dataset = pytreebank.load_sst(path + "trees/")

train_data = dataset['train']
dev_data = dataset['dev']
test_data = dataset['test']

train_list = []
dev_list = []
test_list = []

index = 0

for data in train_data:
  dic = dict()
  dic['label'], dic['text'] = data.to_labeled_lines()[0]
  dic['index'] = index
  index += 1
  train_list.append(dic)

for data in dev_data:
  dic = dict()
  dic['label'], dic['text'] = data.to_labeled_lines()[0]
  dic['index'] = index
  index += 1
  dev_list.append(dic)

for data in test_data:
  dic = dict()
  dic['label'], dic['text'] = data.to_labeled_lines()[0]
  dic['index'] = index
  index += 1
  test_list.append(dic)

all_list = []
all_list.extend(train_list)
all_list.extend(dev_list)
all_list.extend(test_list)

train_index = list(range(len(train_list)))
dev_index = list(range(len(train_list), len(train_list) + len(dev_list)))
test_index = list(range(len(train_list) + len(dev_list), len(all_list)))

index_dict = dict()
index_dict['train'] = train_index
index_dict['valid'] = dev_index
index_dict['test'] = test_index

assert len(set(index_dict['train']).intersection(index_dict['test'])) == 0
assert len(set(index_dict['train']).intersection(index_dict['valid'])) == 0
assert len(set(index_dict['valid']).intersection(index_dict['test'])) == 0

with gzip.open(path + 'index.json.gz', mode='wt') as file:
  json.dump(index_dict, file, ensure_ascii=False)

with gzip.open(path + 'data.json.gz', mode='wt') as file:
  json.dump(all_list, file, ensure_ascii=False)

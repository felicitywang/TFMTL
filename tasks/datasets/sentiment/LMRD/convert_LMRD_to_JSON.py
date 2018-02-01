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
"""Transform original LMRD files into json format"""


# -*- coding: utf-8 -*-

import gzip
import json
import re
from os import listdir
import sys

dir = sys.argv[1]

train_list = []
test_list = []

index = 0

# train pos
path = dir + 'aclImdb/train/pos/'
file_names = listdir(path)
for file_name in file_names:
    file_name_split = re.split("_|\.", file_name)
    with open(path + file_name, "r") as file:
        train_list.append({
            'index': index,
            'id': file_name_split[0],
            'text': file.readline(),
            'score': file_name_split[1],
            'label': "1"})
    index += 1

# train neg
path = dir + 'aclImdb/train/neg/'
file_names = listdir(path)
for file_name in file_names:
    file_name_split = re.split("_|\.", file_name)
    with open(path + file_name, "r") as file:
        train_list.append({
            'index': index,
            'id': file_name_split[0],
            'text': file.readline(),
            'score': file_name_split[1],
            'label': "0"})
    index += 1

# test pos
path = dir + 'aclImdb/test/pos/'
file_names = listdir(path)
for file_name in file_names:
    file_name_split = re.split("_|\.", file_name)
    with open(path + file_name, "r") as file:
        test_list.append({
            'index': index,
            'id': file_name_split[0],
            'text': file.readline(),
            'score': file_name_split[1],
            'label': "1"})
    index += 1

# test neg
path = dir + 'aclImdb/test/neg/'
file_names = listdir(path)
for file_name in file_names:
    file_name_split = re.split("_|\.", file_name)
    (file_name_split)
    with open(path + file_name, "r") as file:
        test_list.append({
            'index': index,
            'id': file_name_split[0],
            'text': file.readline(),
            'score': file_name_split[1],
            'label': "0"})
    index += 1


all_list = []
all_list.extend(train_list)
all_list.extend(test_list)

with gzip.open(dir+'data.json.gz', mode='wt') as file:
    json.dump(all_list, file)

# indices
train_index = list(range(len(train_list)))
test_index = list(range(len(train_list), len(train_list) + len(test_list)))
index = {
    'train': train_index,
    'test': test_index
}
assert len(set(index['train']).intersection(index['test'])) == 0

with gzip.open(dir + 'index.json.gz', mode='wt') as file:
    json.dump(index, file)

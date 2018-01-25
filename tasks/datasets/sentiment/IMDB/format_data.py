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

# -*- coding: utf-8 -*-

# transform IMDB files to json

import gzip
import json
import re
from os import listdir

train_list = []
test_list = []



# train pos
path = 'aclImdb/train/pos/'
file_names = listdir(path)
for file_name in file_names:
    file_name_split = re.split("_|\.", file_name)
    with open(path + file_name, "r") as file:
        train_list.append({
            'id': file_name_split[0],
            'text': file.readline(),
            'score': file_name_split[1],
            'label': "1"})
# train neg
path = 'aclImdb/train/neg/'
file_names = listdir(path)
for file_name in file_names:
    file_name_split = re.split("_|\.", file_name)
    with open(path + file_name, "r") as file:
        train_list.append({
            'id': file_name_split[0],
            'text': file.readline(),
            'score': file_name_split[1],
            'label': "0"})

# test pos
path = 'aclImdb/test/pos/'
file_names = listdir(path)
for file_name in file_names:
    file_name_split = re.split("_|\.", file_name)
    with open(path + file_name, "r") as file:
        test_list.append({
            'id': file_name_split[0],
            'text': file.readline(),
            'score': file_name_split[1],
            'label': "1"})
# test neg
path = 'aclImdb/test/neg/'
file_names = listdir(path)
for file_name in file_names:
    file_name_split = re.split("_|\.", file_name)
    (file_name_split)
    with open(path + file_name, "r") as file:
        test_list.append({
            'id': file_name_split[0],
            'text': file.readline(),
            'score': file_name_split[1],
            'label': "0"})

# # save json
# with open('train.json', 'w') as file:
#     json.dump(train_list, file)
# with open('test.json', 'w') as file:
#     json.dump(test_list, file)

all_list=[]
all_list.extend(train_list)
all_list.extend(test_list)

with gzip.open('data.json.gz', mode='wt') as file:
    json.dump(all_list, file)

# indices
train_index = list(range(len(train_list)))
test_index = list(range(len(train_list), len(train_list) + len(test_list)))
index = {
    'train': train_index,
    'test': test_index
}
assert len(set(index['train']).intersection(index['test'])) == 0

with gzip.open('index.json.gz', mode='wt') as file:
    json.dump(index, file)

# # read json to list
# json_data = open("train.json").read()
# train_list = json.loads(json_data)
# json_data = open("test.json").read()
# test_list = json.loads(json_data)
# json_data = open("data.json").read()
# all_list = json.loads(json_data)

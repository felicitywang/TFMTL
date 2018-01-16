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
"""Transform original SSTb file into json format using pytreebank parser"""

import json

import pytreebank

# path that saves
dataset = pytreebank.load_sst("./original/")

train_data = dataset['train']
dev_data = dataset['dev']
test_data = dataset['test']

train_list = []
dev_list = []
test_list = []

label_list = ["very negative", "negative", "neutral", "positive",
              "very positive"]

for data in train_data:
    dic = dict()
    dic['label'], dic['text'] = data.to_labeled_lines()[0]
    dic['class'] = label_list[dic['label']]
    train_list.append(dic)

for data in dev_data:
    dic = dict()
    dic['label'], dic['text'] = data.to_labeled_lines()[0]
    dic['class'] = label_list[dic['label']]
    dev_list.append(dic)

for data in test_data:
    dic = dict()
    dic['label'], dic['text'] = data.to_labeled_lines()[0]
    dic['class'] = label_list[dic['label']]
    test_list.append(dic)

all_list = []
all_list.extend(train_list)
all_list.extend(dev_list)
all_list.extend(test_list)

train_index = list(range(len(train_list)))
dev_index = list(range(len(train_list), len(train_list) + len(dev_list)))
test_index = list(range(len(train_list) + len(dev_list), len(all_list)))

# pickle.dump(train_index, open("train_index.pickle", "wb"))
# pickle.dump(dev_index, open("dev_index.pickle", "wb"))
# pickle.dump(test_index, open("test_index.pickle", "wb"))
#
# # read index
# train_index = pickle.load(open("train_index.pickle", "rb"))
# dev_index = pickle.load(open("dev_index.pickle", "rb"))
# test_index = pickle.load(open("test_index.pickle", "rb"))

index_dict = dict()
index_dict['train_index'] = train_index
index_dict['dev_index'] = dev_index
index_dict['test_index'] = test_index
with open('index.json', 'w') as file:
    json.dump(index_dict, file)

# with open('train.json', 'w') as file:
#     json.dump(train_list, file)
# with open('dev.json', 'w') as file:
#     json.dump(dev_list, file)
# with open('test.json', 'w') as file:
#     json.dump(test_list, file)
with open('data.json', 'w') as file:
    json.dump(all_list, file)

# # read json to list
# json_data = open("train.json").read()
# train_list = json.loads(json_data)
# json_data = open("dev.json").read()
# dev_list = json.loads(json_data)
# json_data = open("test.json").read()
# test_list = json.loads(json_data)
# json_data = open("data.json").read()
# all_list = json.loads(json_data)

# read json to dict

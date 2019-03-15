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

"""Convert original IMDb files to json format"""

# -*- coding: utf-8 -*-

import gzip
import json
import os
# path that saves
import sys

path = sys.argv[1]

train_list = []
test_list = []

index = 0

with open(os.path.join(path, 'imdb/imdb.neg')) as file:
    for line in file.readlines():
        train_list.append(
            {'index': index,
             'text': line.strip(),
             'label': -1}
        )
        index += 1
    file.close()

with open(os.path.join(path, 'imdb/imdb.pos')) as file:
    for line in file.readlines():
        train_list.append(
            {'index': index,
             'text': line.strip(),
             'label': +1}
        )
        index += 1
    file.close()

all_list = []
all_list.extend(train_list)
all_list.extend(test_list)

train_index = list(range(len(train_list)))
test_index = list(range(len(train_list), len(all_list)))

index_dict = dict()
index_dict['train'] = train_index
index_dict['test'] = test_index

assert len(set(index_dict['train']).intersection(index_dict['test'])) == 0

with gzip.open(os.path.join(path, 'index.json.gz'), mode='wt') as file:
    json.dump(index_dict, file, ensure_ascii=False)

with gzip.open(os.path.join(path, 'data.json.gz'), mode='wt') as file:
    json.dump(all_list, file, ensure_ascii=False)

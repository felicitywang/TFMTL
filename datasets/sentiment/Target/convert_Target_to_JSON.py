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

"""Convert original Target files to json format"""

# -*- coding: utf-8 -*-

import gzip
import json
import os
# path that saves
import sys

dir = sys.argv[1]

train_list = []
test_list = []
index = 0

with open(os.path.join(dir, 'train.raw')) as file:
  lines = file.readlines()
  i = 0
  while i < len(lines):
    line = lines[i].strip()
    target = lines[i + 1].strip()
    label = int(lines[i + 2].strip())
    train_list.append({
      'index': index,
      'old_text': line,
      'text': line.replace('$T$', target),
      'target': target,
      'label': label,
      'start_index': line.find('$T$'),
      'target_length': len(target.split())
    })
    i += 3
    index += 1
  file.close()

with open(os.path.join(dir, 'test.raw')) as file:
  lines = file.readlines()
  i = 0
  while i < len(lines):
    line = lines[i].strip()
    target = lines[i + 1].strip()
    label = int(lines[i + 2].strip())
    test_list.append({
      'index': index,
      'old_text': line,
      'text': line.replace('$T$', target),
      'target': target,
      'label': label,
      'start_index': line.find('$T$'),
      'target_length': len(target.split())
    })
    i += 3
    index += 1
  file.close()


# indices
train_index = list(range(len(train_list)))
test_index = list(range(len(train_list), len(train_list) + len(test_list)))
index_dict = {
  'train': train_index,
  'test': test_index,
}
assert len(set(index_dict['train']).intersection(index_dict['test'])) == 0

with gzip.open(dir + 'index.json.gz', mode='wt') as file:
  json.dump(index_dict, file, ensure_ascii=False)

train_list.append(test_list)

with gzip.open(os.path.join(dir, 'data.json.gz'), mode='wt') as file:
  json.dump(train_list, file, ensure_ascii=False)

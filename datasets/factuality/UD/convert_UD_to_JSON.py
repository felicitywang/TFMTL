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
"""Transform original UD files into json format"""

# -*- coding: utf-8 -*-

import codecs
import gzip
import json
import sys
import conllu


def str2float(s):
  try:
    s = float(s)
  except ValueError:
    pass
  return s


def is_float(e):
  return isinstance(e, float)


dir = sys.argv[1]

FIELDS = ('position', 'form', 'factuality')
index = 0


def parse_data(file_name, starting_index, fields=FIELDS):
  examples = []
  index = starting_index

  with codecs.open(file_name, mode='r', encoding='utf-8') as f:
    data = f.read()
    data = conllu.parse(data, fields=fields)

    for example in data:
      # TODO: clean up sentence? or does that happen downstream?
      sentence = ' '.join([token['form'] for token in example])
      scores = [token['factuality'] for token in example]
      scores = list(map(str2float, scores))
      mask = list(map(is_float, scores))
      judgment_positions = [i for i, x in enumerate(mask) if x is True]

      for position in judgment_positions:
        examples.append({
          'index': index,  # example index
          'text': sentence,
          'token_idx': position,  # which token's factuality is judged
          'label': scores[position],  # factuality judgment
        })
        index += 1

  ending_index = index  # next example will have this index
  index_list = list(range(starting_index, ending_index))
  return examples, ending_index, index_list


train_file = dir + 'train.conll'
dev_file = dir + 'dev.conll'
test_file = dir + 'test.conll'

train_list, index, train_index = parse_data(train_file, index)
dev_list, index, dev_index = parse_data(dev_file, index)
test_list, index, test_index = parse_data(test_file, index)

assert len(set(train_index).intersection(set(dev_index))) == 0
assert len(set(train_index).intersection(set(test_index))) == 0
assert len(set(dev_index).intersection(set(test_index))) == 0

index_dict = {
  'train': train_index,
  'dev': dev_index,
  'test': test_index,
}

data_list = train_list
data_list.extend(dev_list)
data_list.extend(test_list)

with gzip.open(dir + 'index.json.gz', mode='wt') as f:
  json.dump(index_dict, f, ensure_ascii=False)

with gzip.open(dir + 'data.json.gz', mode='wt') as f:
  json.dump(data_list, f, ensure_ascii=False)

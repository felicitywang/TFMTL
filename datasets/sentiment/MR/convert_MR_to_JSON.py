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
"""Transform original Movie Review files into json format"""

# -*- coding: utf-8 -*-

import gzip
import json
import os
import sys

dir = sys.argv[1]

pos_path = 'rt-polaritydata/rt-polarity.pos'
neg_path = 'rt-polaritydata/rt-polarity.neg'

data = []
index = 0

# pos
with open(os.path.join(dir, pos_path), encoding='latin-1') as file:
    for line in file.readlines():
        if not line.strip():
            continue
        data.append({
            'index': index,
            'label': 1,
            'text': line.strip()
        })
        index += 1

# neg
with open(os.path.join(dir, neg_path), encoding='latin-1') as file:
    for line in file.readlines():
        if not line.strip():
            continue
        data.append({
            'index': index,
            'label': 0,
            'text': line.strip()
        })
        index += 1

with gzip.open(dir + 'data.json.gz', mode='wt') as file:
    json.dump(data, file, ensure_ascii=False)

# open for test
with gzip.open(dir + 'data.json.gz', mode='rt') as file:
    data = json.load(file, encoding='utf-8')
    print(len(data))

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

"""Convert original FPGS files to json format"""

# -*- coding: utf-8 -*-

import gzip
import json
import os
# path that saves
import sys

from itertools import islice

path = sys.argv[1]

train_list = []

index = 0

with open(path + "emnlp2015_data/emnlp2015_judgments.txt") as file:
    for line in file.readlines():
        if line.startswith('#'):
            continue
        if not line.strip():
            continue
        line = line.strip().split('\t')
        train_list.append(
            {
                'index': index,
                'subject': line[0],
                'predicate': line[1],
                'mean': float(line[2]),
                'std': float(line[3]),
                'label': round(float(line[2]))
            }
        )
        # print(train_list[index])
        # assert train_list[index]['label'] in [-2, -1, 0, 1, 2]
        index += 1
    file.close()

with gzip.open(os.path.join(path, 'data.json.gz'), mode='wt') as file:
    json.dump(train_list, file, ensure_ascii=False)

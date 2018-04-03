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

"""Convert original POLT files to json format"""

# -*- coding: utf-8 -*-

import gzip
import json
import os
# path that saves
import sys

path = sys.argv[1]

train_list = []

index = 0

label_dict = {'Democratic': 0, 'Republican': 1}

with open(path + "user.politics.time.tsv") as file:
    lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            continue
        line = line.strip()
        while not line.endswith('user'):
            i += 1
            line = (line + ' ' + lines[i]).strip()

        line = line.strip().split('\t')
        if line[0] != line[1]:
            print(line)
        assert line[0] == line[1]
        if line[-1] != 'user':
            print(index, line)
        train_list.append(
            {
                'index': index,
                'userId': line[0],
                'label': label_dict[line[2]],
                'time': line[3],
                'tweetId': line[4],
                'text': line[5]
            }
        )
        i += 1
        index += 1
    file.close()

with gzip.open(os.path.join(path, 'data.json.gz'), mode='wt') as file:
    json.dump(train_list, file, ensure_ascii=False)

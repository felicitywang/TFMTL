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
# See the License for the specific lang governing permissions and
# limitations under the License.
# =============================================================================
"""Read the tsv Turk data and convert to json format

Each domain would have a positive and a negative cut to convert their
average score to binary labels
"""

import math
import gzip
import json
import os
from itertools import product

import pandas as pd

from mtl.util.util import make_dir


def main():
    pos_cuts = [90, 80, 70, 60, 50]
    neg_cuts = [50, 50, 50, 50, 50]

    raw_dir = 'data/raw/TURK'
    json_dir = 'data/json/'

    domains = [
        # 'GOV',
        # 'LIF',
        # 'BUS',
        # 'LAW',
        # 'HEA',
        # 'MIL',
        # 'SPO'
        'REL'
    ]

    for pos, neg, domain in product(pos_cuts, neg_cuts, domains):
        tsvpath = os.path.join(raw_dir, domain + '.tsv')
        df = pd.read_csv(tsvpath, sep='\t')
        data = []
        index = 0
        for ln, item in enumerate(df.to_dict('records')):
            score = float(item['score_mean'])
            if neg <= score <= pos:
                # print(score)
                continue
            # print(score)
            # if math.isnan(score):
            #     print('+1')
            #     continue
            if score < neg:
                label = 0
            else:
                assert score > pos
                label = 1
            data.append({
                'index': index,
                'id': item['id'],
                'text': item['sent'],
                'score': score / 100.0,
                'label': label
            })
            index += 1

        directory = os.path.join(json_dir,
                                 domain + '_turk_' + str(pos) + '_' + str(neg))
        make_dir(directory)
        with gzip.open(
            os.path.join(directory, 'data.json.gz'), mode='wt') as file:
            json.dump(data, file, ensure_ascii=False)

    # open test
    for pos, neg, domain in product(pos_cuts, neg_cuts, domains):
        directory = os.path.join(json_dir,
                                 domain + '_turk_' + str(pos) + '_' + str(neg))
        # print(directory)
        with gzip.open(
            os.path.join(directory, 'data.json.gz'), mode='rt') as file:
            test = json.load(file)
            print('{}: pos={} neg={} all={}'.format(
                directory, len([i for i in test if int(i['label']) == 1]),
                len([i for i in test if int(i['label']) == 0]), len(test)))


if __name__ == '__main__':
    main()

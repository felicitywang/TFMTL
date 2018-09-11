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

"""Read the tsv Turk data and convert to json format"""

import gzip
import json
import os
from itertools import product

import numpy as np
import pandas as pd

from mtl.util.util import make_dir


def main():
    pos_cuts = [90, 80, 70, 60, 50]
    neg_cuts = [50, 50, 50, 50, 50]

    raw_dir = 'data/raw/TURK'
    json_dir = 'data/json/'

    domains = ['GOV', 'LIF', 'BUS', 'LAW', 'HEA', 'MIL', 'SPO']

    for pos, neg, domain in product(pos_cuts, neg_cuts, domains):
        tsvpath = os.path.join(raw_dir, domain + '.tsv')
        df = pd.read_csv(tsvpath, sep='\t')
        data = []
        index = 0
        for item in df.to_dict('records'):
            score = float(item['score_mean'])
            if neg <= score <= pos:
                # print(score)
                continue
            # print(score)
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
        with gzip.open(os.path.join(directory, 'data.json.gz'), mode='wt') as file:
            json.dump(data, file, ensure_ascii=False)

    # open test
    for pos, neg, domain in product(pos_cuts, neg_cuts, domains):
        directory = os.path.join(json_dir,
                                 domain + '_turk_' + str(pos) + '_' + str(neg))
        # print(directory)
        with gzip.open(os.path.join(directory, 'data.json.gz'), mode='rt') as file:
            test = json.load(file)
            print('{}: pos={} neg={} all={}'.format(
                directory,
                len([i for i in test if int(i['label']) == 1]),
                len([i for i in test if int(i['label']) == 0]),
                len(test)))


def get_gold_data(data):
    """Take half pos and half neg as dev/test, return index list"""
    global seed
    np.random.seed(seed)

    pos_list = []
    neg_list = []
    for i, d in enumerate(data):
        if int(d['label']) == 0:
            neg_list.append(i)
        else:
            pos_list.append(i)

    pos = np.random.permutation(np.array(pos_list))
    dev_pos, test_pos = map(list, np.split(pos, [int(len(pos_list) / 2)]))

    neg = np.random.permutation(np.array(neg_list))
    dev_neg, test_neg = map(list, np.split(neg, [int(len(neg_list) / 2)]))

    dev_pos.extend(dev_neg)
    test_pos.extend(test_neg)

    return dev_pos, test_pos


def combine_data(gold_data, syn_data):
    """Combine synthetic data and gold data, get new index_dict"""
    train_index = np.array(list(range(len(syn_data))))
    dev_index, test_index = get_gold_data(gold_data)

    data = []
    data.extend([syn_data[i] for i in train_index])
    data.extend([gold_data[i] for i in dev_index])
    data.extend([gold_data[i] for i in test_index])
    for index, item in enumerate(data):
        item['index'] = index
    index_dict = {
        'train': list(range(len(train_index))),
        'valid': list(
            range(len(train_index), len(train_index) + len(dev_index))),
        'test': list(range(len(train_index) + len(dev_index),
                           len(train_index) + len(dev_index) + len(test_index)))
    }

    return data, index_dict


if __name__ == '__main__':
    main()

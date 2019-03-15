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
# =============================================================================
"""Combine different datasets into one's splits

Usage:
    python combine_data_splits.py --train training_split_suffixes --valid validation_data_split
e.g.
    python combine_data_splits.py --train syn_p1000r1000 --valid gold_one gold_oracle
"""

import argparse as ap
import gzip
import json
import os

from mtl.util.util import make_dir

base_dir = 'data/json/'


def parse_args():
    p = ap.ArgumentParser()
    p.add_argument(
        '--train_suffixes',
        nargs='+',
        type=str,
        required=True,
        help='suffixes of the names of the datasets to be the train split')
    p.add_argument(
        '--valid_suffixes',
        nargs='+',
        type=str,
        default=[],
        help='suffixes of the names of the datasets to be used as the valid split')
    p.add_argument(
        '--domain',
        type=str,
        help='Name of the domain')
    # test currently not supported
    # p.add_argument('--test', type=str, nargs='?', required=False
    #                help='Name of the dataset to be used as the test split')
    return p.parse_args()


def get_data(domain, suffix):
    path = os.path.join(base_dir, domain + '_' + suffix, 'data.json.gz')
    with gzip.open(path, 'rt') as file:
        data = json.load(file)
    return data


def main():
    args = parse_args()

    train_data = []
    valid_data = []
    for train_suffix in args.train_suffixes:
        train_data.extend(get_data(args.domain, train_suffix))

    for valid_suffix in args.valid_suffixes:
        valid_data.extend(get_data(args.domain, valid_suffix))

    index_dict = {
        'train':
            list(range(len(train_data))),
        'valid':
            list(range(len(train_data),
                       len(train_data) + len(valid_data))),
        'test': []
    }

    suffix = '_'.join(args.train_suffixes)
    if args.valid_suffixes:
        suffix = 'train_' + suffix + '_valid_' + ' '.join(args.valid_suffixes)

    dout = os.path.join(
        base_dir,
        args.domain + '_' + suffix
    )
    make_dir(dout)

    print(dout)
    print('train:', len(train_data))
    print('valid:', len(valid_data))

    # continue

    data = train_data
    data.extend(valid_data)

    with gzip.open(os.path.join(dout, 'data.json.gz'), mode='wt') as file:
        json.dump(data, file, ensure_ascii=False)

    if args.valid_suffixes:
        with gzip.open(os.path.join(dout, 'index.json.gz'), mode='wt') as file:
            json.dump(index_dict, file, ensure_ascii=False)


if __name__ == '__main__':
    main()

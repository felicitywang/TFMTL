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
    python combine_data_splits.py --train training_split_suffix --valid validation_data_split
e.g.
    python combine_data_splits.py --train syn_p1000r1000 --valid gold_one
"""

import argparse as ap
import gzip
import json
import os

from mtl.util.util import make_dir
from tqdm import tqdm

base_dir = 'data/json/'

DOMAINS = [
  'GOV',
  'LIF',
  'BUS',
  'LAW',
  # 'SPO',
  'HEA',
  'MIL'
]


def parse_args():
  p = ap.ArgumentParser()
  p.add_argument(
    '--train_suffix',
    type=str,
    required=True,
    help='Suffix of the name of the dataset to be the train split')
  p.add_argument(
    '--valid_suffix',
    type=str,
    required=True,
    help='Suffix of the ame of the dataset to be used as the valid split')
  # test currently not supported
  # p.add_argument('--test', type=str, nargs='?', required=False
  #                help='Name of the dataset to be used as the test split')
  return p.parse_args()


def main():
  args = parse_args()

  for domain in tqdm(DOMAINS):
    train_path = os.path.join(base_dir, domain + '_' + args.train_suffix,
                              'data.json.gz')
    valid_path = os.path.join(base_dir, domain + '_' + args.valid_suffix,
                              'data.json.gz')

    with gzip.open(train_path, 'rt') as file:
      train_data = json.load(file)
    with gzip.open(valid_path, 'rt') as file:
      valid_data = json.load(file)

    index_dict = {
      'train':
        list(range(len(train_data))),
      'valid':
        list(range(len(train_data),
                   len(train_data) + len(valid_data))),
      'test': []
    }

    data = train_data
    data.extend(valid_data)

    dout = os.path.join(
      base_dir, domain + '_train_' + args.train_suffix + '_valid_' +
                args.valid_suffix)
    make_dir(dout)

    with gzip.open(os.path.join(dout, 'data.json.gz'), mode='wt') as file:
      json.dump(data, file, ensure_ascii=False)
    with gzip.open(os.path.join(dout, 'index.json.gz'), mode='wt') as file:
      json.dump(index_dict, file, ensure_ascii=False)


if __name__ == '__main__':
  main()

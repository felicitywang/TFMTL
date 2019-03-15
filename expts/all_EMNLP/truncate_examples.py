#! /usr/bin/env python3

# Copyright 2018 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Backup the original json file and truncate the new one to maximum
sequence length"""
import functools
import gzip
import json
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

from mtl.util.data_prep import ruder_tokenizer


def main():
    tokenizer = functools.partial(ruder_tokenizer, preserve_case=False)
    # dataset = 'FNC-1'
    # max_len = 100

    dataset = sys.argv[1]
    max_len = int(sys.argv[2])

    truncated_num_seq1 = 0
    truncated_num_seq2 = 0

    # backup
    if not Path('data/json/' + dataset + '.bak').exists():
        shutil.copytree('data/json/' + dataset, 'data/json/' + dataset + '.bak')

    with gzip.open('data/json/' + dataset + '.bak' + '/data.json.gz',
                   mode='rt') as file:
        data_list = json.load(file, encoding='utf-8')
    for data in tqdm(data_list):
        # data['seq1'] = ' '.join(data['seq1'].split()[:max_len])
        # data['seq2'] = ' '.join(data['seq2'].split()[:max_len])
        seq1_list = tokenizer(data['seq1'])
        seq2_list = tokenizer(data['seq2'])
        if len(seq1_list) > max_len:
            truncated_num_seq1 += 1
        if len(seq2_list) > max_len:
            truncated_num_seq2 += 1
        data['seq1'] = ' '.join(tokenizer(data['seq1'])[:max_len])
        data['seq2'] = ' '.join(tokenizer(data['seq2'])[:max_len])

    print('seq1 truncated num:', truncated_num_seq1)
    print('seq2 truncated num:', truncated_num_seq2)

    with gzip.open('data/json/' + dataset + '/data.json.gz', mode='wt') as file:
        json.dump(data_list, file, ensure_ascii=False)

    # with gzip.open('data/json/' + dataset + '/data.json.gz', mode='rt') as file:
    #   data_list = json.load(file, encoding='utf-8')
    # for data in tqdm(data_list):
    #   assert len(data['seq1'].split()) <= max_len
    #   assert len(data['seq2'].split()) <= max_len


if __name__ == '__main__':
    main()

#! /usr/bin/env python

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
# ============================================================================

import json
import os
import sys

from mtl.util.dataset import merge_dict_write_tfrecord, \
  merge_pretrain_write_tfrecord
from mtl.util.util import make_dir

# TODO separate args file for each dataset?

with open('args_merged.json', 'rt') as file:
  args_merged = json.load(file)

json_dirs = [os.path.join('data/json/', argv) for argv in sys.argv[1:]]
print(json_dirs)

tfrecord_dir = "data/tf/merged/"
datasets = sorted(sys.argv[1:])
for argv in datasets[:-1]:
  tfrecord_dir += argv + "_"
tfrecord_dir += datasets[-1] + '/'

if 'pretrained_file' not in args_merged or not args_merged['pretrained_file']:
  tfrecord_dir = os.path.join(tfrecord_dir,
                              "min_" + str(args_merged['min_frequency']) + \
                              "_max_" + str(args_merged['max_frequency']) + \
                              "_vocab_" + str(args_merged['max_vocab_size']))
  tfrecord_dirs = [os.path.join(tfrecord_dir, argv) for argv in sys.argv[1:]]
  for i in tfrecord_dirs:
    make_dir(i)
  merge_dict_write_tfrecord(json_dirs=json_dirs,
                            tfrecord_dirs=tfrecord_dirs,
                            merged_dir=tfrecord_dir,
                            max_document_length=args_merged[
                              'max_document_length'],
                            max_vocab_size=args_merged['max_vocab_size'],
                            min_frequency=args_merged['min_frequency'],
                            max_frequency=args_merged['max_frequency'],
                            text_field_names=args_merged['text_field_names'],
                            label_field_name=args_merged['label_field_name'],
                            train_ratio=args_merged['train_ratio'],
                            valid_ratio=args_merged['valid_ratio'],
                            tokenizer_=args_merged['tokenizer'],
                            subsample_ratio=args_merged['subsample_ratio'],
                            padding=args_merged['padding'],
                            write_bow=args_merged['write_bow'],
                            write_tfidf=args_merged['write_tfidf'])
else:
  vocab_path = args_merged['pretrained_file']
  vocab_dir = os.path.dirname(vocab_path)
  vocab_name = os.path.basename(vocab_path)
  tfrecord_dir = os.path.join(tfrecord_dir, vocab_name[:vocab_name.find(
    '.txt')])
  tfrecord_dirs = [os.path.join(tfrecord_dir, argv) for argv in sys.argv[1:]]
  for i in tfrecord_dirs:
    make_dir(i)
  combine_pretrain_train = False
  if 'combine_pretrain_train' in args_merged:
    combine_pretrain_train = args_merged['combine_pretrain_train']
  merge_pretrain_write_tfrecord(json_dirs=json_dirs,
                                tfrecord_dirs=tfrecord_dirs,
                                merged_dir=tfrecord_dir,
                                vocab_dir=vocab_dir,
                                vocab_name=vocab_name,
                                text_field_names=args_merged[
                                  'text_field_names'],
                                label_field_name=args_merged[
                                  'label_field_name'],
                                max_document_length=args_merged[
                                  'max_document_length'],
                                # max_vocab_size=args_merged['max_vocab_size'],
                                # min_frequency=args_merged['min_frequency'],
                                # max_frequency=args_merged['max_frequency'],
                                train_ratio=args_merged['train_ratio'],
                                valid_ratio=args_merged['valid_ratio'],
                                subsample_ratio=args_merged['subsample_ratio'],
                                padding=args_merged['padding'],
                                write_bow=args_merged['write_bow'],
                                write_tfidf=args_merged['write_tfidf'],
                                tokenizer_=args_merged['tokenizer'],
                                combine_pretrain_train=combine_pretrain_train)

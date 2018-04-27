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

from mtl.util.dataset import Dataset

if len(sys.argv) == 2:
  args_name = 'args_' + sys.argv[1] + '.json'
else:
  args_name = sys.argv[2]

with open(args_name, 'rt') as file:
  args_single = json.load(file)

json_dir = "data/json/" + sys.argv[1]

tfrecord_dir = os.path.join("data/tf/single/", sys.argv[1])

if 'pretrained' not in args_single or not args_single['pretrained']:
  tfrecord_dir = os.path.join(tfrecord_dir,
                              "min_" + str(args_single['min_frequency']) + \
                              "_max_" + str(args_single['max_frequency']) + \
                              "_vocab_" + str(args_single['max_vocab_size']))
  dataset = Dataset(json_dir=json_dir,
                    tfrecord_dir=tfrecord_dir,
                    vocab_dir=tfrecord_dir,
                    text_field_names=args_single['text_field_names'],
                    label_field_name=args_single['label_field_name'],
                    max_document_length=args_single['max_document_length'],
                    max_vocab_size=args_single['max_vocab_size'],
                    min_frequency=args_single['min_frequency'],
                    max_frequency=args_single['max_frequency'],
                    train_ratio=args_single['train_ratio'],
                    valid_ratio=args_single['valid_ratio'],
                    subsample_ratio=args_single['subsample_ratio'],
                    padding=args_single['padding'],
                    write_bow=args_single['write_bow'],
                    write_tfidf=args_single['write_tfidf'],
                    tokenizer_=args_single['tokenizer'],
                    generate_basic_vocab=False,
                    vocab_given=False,
                    generate_tf_record=True)
else:
  vocab_path = args_single['pretrained']
  vocab_dir = os.path.dirname(vocab_path)
  vocab_name = os.path.basename(vocab_path)
  tfrecord_dir = os.path.join(tfrecord_dir, vocab_name[:vocab_name.find(
    '.txt')])
  combine_pretrain_train = False
  if 'combine_pretrain_train' in args_single:
    combine_pretrain_train = args_single['combine_pretrain_train']
  dataset = Dataset(json_dir=json_dir,
                    tfrecord_dir=tfrecord_dir,
                    vocab_given=True,
                    vocab_dir=vocab_dir,
                    vocab_name=vocab_name,
                    text_field_names=args_single['text_field_names'],
                    label_field_name=args_single['label_field_name'],
                    # max_document_length=args_single['max_document_length'],
                    # max_vocab_size=args_single['max_vocab_size'],
                    # min_frequency=args_single['min_frequency'],
                    # max_frequency=args_single['max_frequency'],
                    train_ratio=args_single['train_ratio'],
                    valid_ratio=args_single['valid_ratio'],
                    subsample_ratio=args_single['subsample_ratio'],
                    padding=args_single['padding'],
                    write_bow=args_single['write_bow'],
                    write_tfidf=args_single['write_tfidf'],
                    tokenizer_=args_single['tokenizer'],
                    generate_basic_vocab=False,
                    generate_tf_record=True,
                    combine_pretrain_train=combine_pretrain_train)

with open(os.path.join(tfrecord_dir, 'vocab_size.txt'), 'w') as f:
  f.write(str(dataset.vocab_size))

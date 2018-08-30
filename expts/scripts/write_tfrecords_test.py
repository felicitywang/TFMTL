# ! /usr/bin/env python

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

"""Write test.tf with given test.json.gz"""
import json
import os
import sys

from docutils.io import InputError

from mtl.util.dataset import Dataset

if len(sys.argv) != 5:
  raise InputError(
    "Usage: python write_tfrecords_test.py args_test_json_path "
    "test_json_dir tfrecord_dir vocab_dir")

# TODO REFACTOR!!!


args_test_path = sys.argv[1]
json_dir = sys.argv[2]
tfrecord_dir = sys.argv[3]
vocab_dir = sys.argv[4]

# find the used arguments
if os.path.exists(os.path.join(os.path.abspath(vocab_dir), 'args.json')):
  args_path = os.path.join(os.path.abspath(vocab_dir), 'args.json')
else:
  args_path = os.path.join(vocab_dir, os.listdir(vocab_dir)[0], 'args.json')

with open(args_path) as file:
  args_used = json.load(file)

with open(args_test_path)as file:
  args = json.load(file)

dataset = Dataset(

  # keep consistent with the training datasets
  max_document_length=args_used['max_document_length'],
  max_vocab_size=args_used['max_vocab_size_allowed'],
  min_frequency=args_used['min_frequency'],
  max_frequency=args_used['max_frequency'],
  padding=args_used.get('padding', args['padding']),
  write_bow=args_used.get('write_bow', args['write_bow']),
  write_tfidf=args_used.get('write_tfidf', args['write_tfidf']),
  tokenizer_=args_used.get('tokenizer', args['tokenizer']),
  preproc=args_used.get('preproc', args.get('preproc', True)),
  vocab_all=args_used.get('vocab_all', args.get('vocab_all', False)),

  # may be different
  text_field_names=args['text_field_names'],
  label_field_name=args['label_field_name'],

  # test split only
  train_ratio=0.0,
  valid_ratio=0.0,

  # default in test mode
  json_dir=json_dir,
  tfrecord_dir=tfrecord_dir,
  vocab_dir=vocab_dir,
  generate_basic_vocab=False,
  vocab_given=True,
  vocab_name='vocab_v2i.json',
  generate_tf_record=True
)

# with open(tfrecord_dir + 'vocab_size.txt', 'w') as f:
#   f.write(str(dataset.vocab_size))

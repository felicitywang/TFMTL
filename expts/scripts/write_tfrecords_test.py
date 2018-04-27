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

from mtl.util.dataset import Dataset

if len(sys.argv) != 6:
  print(
    "Usage: python write_tfrecords_test.py test_json_dir tfrecord_dir "
    "vocab_dir args_test_json_path")

# dataset_args_path = sys.argv[1]

json_dir = sys.argv[1]
tfrecord_dir = sys.argv[2]
vocab_dir = sys.argv[3]
args_test_path = sys.argv[4]

# compute max document length
# max document length should be the max(max_document_lengths) for each
# dataset used to generate the vocabulary, their max_document_length happens
#  to be in vocab_dir/DATASET/args.json
# otherwise, vocabulary comes from a single dataset, whose args.jon lies in
# vocab_dir
args_paths = [os.path.join(vocab_dir, folder, 'args.json') for folder in
              os.listdir(vocab_dir) if
              os.path.isdir(os.path.join(vocab_dir, folder))]

if len(args_paths) == 0:
  args_paths = [os.path.join(vocab_dir, 'args.json')]
max_document_lengths = [
  json.load(open(args_path, 'r'))['max_document_length']
  for args_path in args_paths]

with open(args_test_path)as file:
  args_test = json.load(file)

dataset = Dataset(json_dir=json_dir,
                  tfrecord_dir=tfrecord_dir,
                  vocab_dir=vocab_dir,
                  vocab_name='vocab_v2i.json',
                  max_document_length=max(max_document_lengths),
                  text_field_names=args_test['text_field_names'],
                  label_field_name=args_test['label_field_name'],
                  padding=args_test['padding'],
                  write_bow=args_test['write_bow'],
                  write_tfidf=args_test['write_tfidf'],
                  tokenizer_=args_test['tokenizer'],
                  train_ratio=0.0,
                  valid_ratio=0.0,
                  generate_basic_vocab=False,
                  vocab_given=True,
                  generate_tf_record=True)

# with open(tfrecord_dir + 'vocab_size.txt', 'w') as f:
#   f.write(str(dataset.vocab_size))

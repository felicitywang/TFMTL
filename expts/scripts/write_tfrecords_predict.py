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

if len(sys.argv) != 6:
  print("Usage: python write_tfrecords_predict.py DATASET_NAME "
        "predict_json_path predict_tf_path tfrecord_dir vocab_dir")

dataset_args_path = sys.argv[1]
predict_json_path = sys.argv[2]
predict_tf_path = sys.argv[3]
tfrecord_dir = sys.argv[4]
vocab_dir = sys.argv[5]

# args_DATASET.json or args_merged.json which has min_freq, max_freq,
# max_document_length etc. information, which are used to further build
# vocabulary
with open(dataset_args_path, 'rt') as file:
  args_predict = json.load(file)
  file.close()

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

dataset = Dataset(json_dir=None,
                  tfrecord_dir=tfrecord_dir,
                  vocab_dir=vocab_dir,
                  max_document_length=max(max_document_lengths),
                  text_field_names=args_predict['text_field_names'],
                  label_field_name=args_predict['label_field_name'],
                  padding=args_predict['padding'],
                  write_bow=args_predict['write_bow'],
                  write_tfidf=args_predict['write_tfidf'],
                  tokenizer_=args_predict['tokenizer'],
                  generate_basic_vocab=False,
                  vocab_given=True,
                  vocab_name='vocab_v2i.json',
                  generate_tf_record=True,
                  predict_mode=True,
                  predict_json_path=predict_json_path,
                  predict_tf_path=predict_tf_path)

# with open(tfrecord_dir + 'vocab_size.txt', 'w') as f:
#   f.write(str(dataset.vocab_size))

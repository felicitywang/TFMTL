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

# get max document length from the processed dataset folder
tf_args_path = os.path.join(tfrecord_dir, 'args.json')
with open(tf_args_path) as file:
  args = json.load(file)
  max_document_length = args['max_document_length']

dataset = Dataset(json_dir=None,
                  tfrecord_dir=tfrecord_dir,
                  vocab_dir=vocab_dir,
                  max_document_length=max_document_length,
                  padding=args_predict['padding'],
                  write_bow=args_predict['write_bow'],
                  write_tfidf=args_predict['write_tfidf'],
                  generate_basic_vocab=False,
                  vocab_given=True,
                  vocab_name='vocab_v2i.json',
                  generate_tf_record=True,
                  predict_mode=True,
                  predict_json_path=predict_json_path,
                  predict_tf_path=predict_tf_path)

# with open(tfrecord_dir + 'vocab_size.txt', 'w') as f:
#   f.write(str(dataset.vocab_size))

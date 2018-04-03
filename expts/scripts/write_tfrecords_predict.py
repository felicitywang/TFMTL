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

if len(sys.argv) != 5:
  print("Usage: python write_tfrecords_predict.py DATASET_NAME "
        "predict_json_path predict_tf_path tf_record_dir")

with open('args_' + sys.argv[1] + '.json', 'rt') as file:
  args_predict = json.load(file)
  file.close()

predict_json_path = sys.argv[2]
predict_tf_path = sys.argv[3]

json_dir = sys.argv[1]

tfrecord_dir = sys.argv[4]
args_json_path = os.path.join(tfrecord_dir, 'args.json')
with open(args_json_path) as file:
  args = json.load(file)
  max_document_length = args['max_document_length']

dataset = Dataset(json_dir=json_dir,
                  tfrecord_dir=tfrecord_dir,
                  vocab_dir=tfrecord_dir,
                  max_document_length=max_document_length,
                  padding=args_predict['padding'],
                  write_bow=args_predict['write_bow'],
                  write_tfidf=args_predict['write_tfidf'],
                  generate_basic_vocab=False,
                  vocab_given=True,
                  generate_tf_record=True,
                  predict_mode=True,
                  predict_json_path=predict_json_path,
                  predict_tf_path=predict_tf_path)

# with open(tfrecord_dir + 'vocab_size.txt', 'w') as f:
#   f.write(str(dataset.vocab_size))


# TODO from plain text file

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
import sys

from mtl.util.dataset import Dataset
from mtl.util.util import make_dir

with open('args_' + sys.argv[1] + '.json', 'rt') as file:
  args_predict = json.load(file)
  file.close()

json_dir = "data/json/" + sys.argv[1]

tfrecord_dir = "data/tf/predict/"
tfrecord_dir += sys.argv[1] + "/"
tfrecord_dir += "min_" + str(args_predict['min_frequency']) + \
                "_max_" + str(args_predict['max_frequency']) + "/"
make_dir(tfrecord_dir)

dataset = Dataset(json_dir=json_dir,
                  tfrecord_dir=tfrecord_dir,
                  vocab_dir=tfrecord_dir,
                  max_document_length=args_predict['max_document_length'],
                  padding=args_predict['padding'],
                  write_bow=args_predict['write_bow'],
                  write_tfidf=args_predict['write_tfidf'],
                  generate_basic_vocab=False,
                  vocab_given=True,
                  generate_tf_record=True,
                  predict_mode=True,
                  # TODO predict file name from command line
                  predict_file_name='predict.json.gz')

with open(tfrecord_dir + 'vocab_size.txt', 'w') as f:
  f.write(str(dataset.vocab_size))

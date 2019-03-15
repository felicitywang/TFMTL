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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from collections import defaultdict

import numpy as np
import tensorflow as tf
from google.protobuf.json_format import MessageToJson


def get_num_records(tf_record_filename):
    c = 0
    for record in tf.python_io.tf_record_iterator(tf_record_filename):
        c += 1
    return c


def get_empirical_label_prior(tf_record_filename, label_key="label"):
    freq = defaultdict(int)
    for record in tf.python_io.tf_record_iterator(tf_record_filename):
        jsonMessage = MessageToJson(tf.train.Example.FromString(record))
        d = json.loads(jsonMessage)
        label = int(
            d['features']['feature'][label_key]['int64List']['value'][0])
        freq[label] += 1
    N = float(sum(freq.values()))
    p = np.zeros(len(freq))
    for k, c in freq.items():
        p[k] = freq[k] / N
    return p

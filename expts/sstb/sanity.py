#! /usr/bin/env python

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Copyright 2018 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import json
import os
import threading
import datetime

import model
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import variables
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.saved_model import signature_constants as sig_constants

tf.logging.set_verbosity(tf.logging.INFO)

import model
import argparse as ap

if __name__ == "__main__":
    p = ap.ArgumentParser()
    p.add_argument('path')
    args = p.parse_args()

    # Create a new graph and specify that as default
    with tf.Graph().as_default():
        batch = model.input_fn(args.path,
                               num_epochs=1, shuffle=False,
                               batch_size=32)

        with tf.Session() as sess:
            batch_v = sess.run(batch)
            print(batch_v)

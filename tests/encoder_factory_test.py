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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from mtl.util.encoder_factory import build_encoders

class EncoderTests(tf.test.TestCase):
  def test_template(self):
    with self.test_session() as sess:
      # TODO: create fully tied encoders and check that emb1 == emb2, extr1 == extr2, etc.
      # TODO: create fully untied encoders and check equalities/inequalities
      # TODO: create tied embeddings, untied extractors ...
      # TODO: create untied embeddings, tied extractors ...
      pass

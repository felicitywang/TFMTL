# Copyright 2017 Johns Hopkins University. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from mlvae import SymbolTable

def _isnum(x):
  try:
    int(x)
    return True
  except:
    return False


def get_type(w):
  if _isnum(w):
    return "num"
  else:
    return "word"


class SymbolTableTest(tf.test.TestCase):
  # Simple observations
  train = "one two two two three three four four four four"
  valid = "one two two five five"

  def init_table(self):
    table = SymbolTable()
    for token in self.train.split():
      table.add(token)
    return table

  def test_basic(self):
    table = self.init_table()
    table.freeze()
    self.assertEqual(table.value_to_index, {
      'four': 0,
      'two': 1,
      'three': 2,
      'one': 3
    })
    train_ids = [table.idx(w) for w in SymbolTableTest.train.split()]
    self.assertEqual([3, 1, 1, 1, 2, 2, 0, 0, 0, 0], train_ids)
    self.assertEqual(SymbolTableTest.train, " ".join(table.val(train_ids)))

  def test_unk(self):
    table = self.init_table()
    oov='<oov>'
    table.freeze(unk=oov)
    self.assertEqual(table.value_to_index, {
      'four': 0,
      'two': 1,
      'three': 2,
      'one': 3,
      oov: 4
    })
    valid_ids = [table.idx(w) for w in SymbolTableTest.valid.split()]
    self.assertEqual([3, 1, 1, 4, 4], valid_ids)
    self.assertEqual("one two two <oov> <oov>", " ".join(table.val(valid_ids)))

  def test_min_freq(self):
    table = self.init_table()
    oov='<oov>'
    table.freeze(unk=oov, min_freq=2)
    self.assertEqual(table.value_to_index, {
      'four': 0,
      'two': 1,
      'three': 2,
      oov: 3
    })
    valid_ids = [table.idx(w) for w in SymbolTableTest.valid.split()]
    self.assertEqual([3, 1, 1, 3, 3], valid_ids)
    self.assertEqual("<oov> two two <oov> <oov>", " ".join(table.val(valid_ids)))

  def test_max_size(self):
    table = self.init_table()
    oov='<oov>'
    table.freeze(unk=oov, max_size=3)
    self.assertEqual(table.value_to_index, {
      'four': 0,
      'two': 1,
      'three': 2,
      oov: 3
    })
    valid_ids = [table.idx(w) for w in SymbolTableTest.valid.split()]
    self.assertEqual([3, 1, 1, 3, 3], valid_ids)
    self.assertEqual("<oov> two two <oov> <oov>", " ".join(table.val(valid_ids)))

  def test_max_size_and_min_freq(self):
    table = self.init_table()
    oov='<oov>'
    table.freeze(unk=oov, max_size=3, min_freq=2)
    self.assertEqual(table.value_to_index, {
      'four': 0,
      'two': 1,
      'three': 2,
      oov: 3
    })
    valid_ids = [table.idx(w) for w in SymbolTableTest.valid.split()]
    self.assertEqual([3, 1, 1, 3, 3], valid_ids)
    self.assertEqual("<oov> two two <oov> <oov>", " ".join(table.val(valid_ids)))


if __name__ == "__main__":
  tf.test.main()

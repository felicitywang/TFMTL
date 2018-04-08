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

"""Test cases from sklearn.metrics examples"""

import tensorflow as tf

from mtl.util.metrics import accuracy_score, accurate_number, recall_macro, \
  f1_macro, mae_macro


class MetricTest(tf.test.TestCase):
  def test_accuracy(self):
    y_trues = [0, 1, 2, 3]
    y_preds = [0, 2, 1, 3]
    labels = [0, 1, 2, 3]
    self.assertEqual(
      accuracy_score(y_trues=y_trues, y_preds=y_preds, labels=labels),
      0.5
    )

  def test_accurate_number(self):
    y_trues = [0, 1, 2, 3]
    y_preds = [0, 2, 1, 3]
    self.assertAllEqual(
      accurate_number(y_trues=y_trues, y_preds=y_preds),
      2
    )

  def test_recall_macro(self):
    y_trues = [0, 1, 2, 0, 1, 2]
    y_preds = [0, 2, 1, 0, 0, 1]
    labels = [0, 1, 2]
    self.assertAlmostEqual(
      recall_macro(y_trues=y_trues, y_preds=y_preds, labels=labels),
      1 / 3
    )

  def test_f1_macro(self):
    y_trues = [0, 1, 2, 0, 1, 2]
    y_preds = [0, 2, 1, 0, 0, 1]
    labels = [0, 1, 2]
    self.assertAllEqual(
      f1_macro(y_trues=y_trues, y_preds=y_preds, labels=labels),
      4 / 15
    )

  def mae_macro(self):
    y_trues = [3, -0.5, 2, 7]
    y_preds = [2.5, 0.0, 2, 8]
    self.assertEqual(
      mae_macro(y_trues=y_trues, y_preds=y_preds, labels=None),
      0.5
    )


if __name__ == '__main__':
  tf.test.main()

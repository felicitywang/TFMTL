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
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange
from scipy.optimize import linear_sum_assignment


def aligned_accuracy(gold_labels, guess_labels):
  gold_label_set = set(gold_labels)

  # Number of examples
  N = len(gold_labels)
  assert len(guess_labels) == N
  assert 0 in gold_label_set, "this assumes the labels are 0-indexed"

  # Number of classes
  K = len(gold_label_set)

  # Create confusion matrix
  C = np.zeros((K, K))
  for k1 in xrange(K):
    for k2 in xrange(K):
      c = 0
      for j in xrange(N):
        if guess_labels[j] == k1 and gold_labels[j] == k2:
          c += 1
      C[k1][k2] -= c

  # Find best assignment
  row_ind, col_ind = linear_sum_assignment(C)

  for j in xrange(N):
    guess_labels[j] = col_ind[guess_labels[j]]

  total_correct = sum(x[0] == x[1] for x in zip(gold_labels, guess_labels))
  return float(total_correct) / float(N)

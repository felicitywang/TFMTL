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

# -*- coding: utf-8 -*-


"""Different evaluation metric APIs from sklearn.metrics

accuracy_score:   accuracy classification score
accurate_number:  number of correctly predicted labels
f1_macro:         macro-averaged(unweighted mean) F1 score
mae_macro:        macro-averaged(unweighted mean) mean absolute error
recall_macro:     macro-averaged(unweighted mean) recall score

More details see sklearn documentation
http://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation
"""

import sklearn.metrics
import numpy as np


def accuracy_score(y_trues, y_preds, labels, topics):
  return sklearn.metrics.accuracy_score(y_true=y_trues,
                                        y_pred=y_preds,
                                        normalize=True  # return fraction
                                        )


def accurate_number(y_trues, y_preds, labels, topics):
  return sklearn.metrics.accuracy_score(y_true=y_trues,
                                        y_pred=y_preds,
                                        normalize=False  # return number
                                        )


def f1_macro(y_trues, y_preds, labels, topics):
  """
  macro-averaged (unweighted mean) f1 score of all classes

  :param y_trues: list of ground truth labels
  :param y_preds: list of predicted labels
  :param labels: labels for each class in a list, must specify
  :return: float
  """
  assert labels is not None
  return sklearn.metrics.f1_score(y_true=y_trues,
                                  y_pred=y_preds,
                                  labels=labels,
                                  average='macro'
                                  )


def f1_pos_neg_macro(y_trues, y_preds, labels, topics):
  assert labels is not None
  f1_scores = sklearn.metrics.f1_score(y_true=y_trues,
                                       y_pred=y_preds,
                                       labels=labels,
                                       average=None
                                       )
  # Assumes that POS and NEG-like labels are in positions 0 and 1
  # and that any NONE-like labels are in position 2 onwards
  # (and will be ignored)
  return np.mean([f1_scores[0], f1_scores[1]])


def mae_macro(y_trues, y_preds, labels, topics):
  """
  macro-averaged (unweighted mean) over topics of mean absolute error

  :param y_trues: list of ground truth labels
  :param y_preds: list of predicted labels
  :return: float
  """
  if len(topics) == 0:
    return float('inf')

  topics_set = set(topics)

  preds = list(zip(*[y_trues, y_preds, topics]))
  preds_by_topic = dict()
  for topic in topics_set:
    preds_by_topic[topic] = []

  # group predictions by topic
  for pred in preds:
    preds_by_topic[pred[2]].append(pred)

  maes = dict()
  for topic, preds in preds_by_topic.items():
    y_true = [p[0] for p in preds]
    y_pred = [p[1] for p in preds]

    if labels:
      # macro-average over labels as well as over topics
      # following code released for: https://arxiv.org/abs/1802.09913
      tmp_maes = []
      for label in labels:
        true_pred_pairs = [(y_t, y_p) for y_t, y_p in
                           zip(*[y_true, y_pred])
                           if y_t == label]
        if len(true_pred_pairs) == 0:
          continue
        tmp_y_true, tmp_y_pred = zip(*true_pred_pairs)

        mean_absolute_error = sklearn.metrics.mean_absolute_error
        tmp_mae = mean_absolute_error(y_true=tmp_y_true,
                                      y_pred=tmp_y_pred,
                                      multioutput='uniform_average')
        tmp_maes.append(tmp_mae)

      mae = np.mean(tmp_maes)
    else:
      # macro-average over topics but not labels
      mae = sklearn.metrics.mean_absolute_error(y_true=y_true,
                                                y_pred=y_pred,
                                                multioutput='uniform_average')
    maes[topic] = mae

  # simple mean of maes over topics
  return sum(maes.values()) / len(maes.values())


def neg_mae_macro(y_trues, y_preds, labels, topics):
  """
  As for absolute error, lower is better
  Thus use negative value in order to share the same interface when tuning
  dev data with other metrics
  """
  return -mae_macro(y_trues, y_preds, labels, topics)


def recall_macro(y_trues, y_preds, labels, topics):
  """
  macro-averaged (unweighted mean) over topics of recall score of all classes

  :param y_trues: list of ground truth labels
  :param y_preds: list of predicted labels
  :param labels: labels for each class in a list, must specify
  :return: float
  """
  if len(topics) == 0:
    return float('-inf')

  topics_set = set(topics)
  # print('{} topics'.format(len(topics_set)))

  preds = list(zip(*[y_trues, y_preds, topics]))
  preds_by_topic = dict()
  for topic in topics_set:
    preds_by_topic[topic] = []

  # group predictions by topic
  for pred in preds:
    preds_by_topic[pred[2]].append(pred)

  recalls = dict()
  for topic, preds in preds_by_topic.items():
    y_true = [p[0] for p in preds]
    y_pred = [p[1] for p in preds]

    r = sklearn.metrics.recall_score(y_true=y_true,
                                     y_pred=y_pred,
                                     average='macro')
    recalls[topic] = r

  # for topic, recall in recalls.items():
  #   print('{}: recall={}'.format(topic, recall))

  # simple mean of recalls over topics
  return sum(recalls.values()) / len(recalls.values())


def metric2func(metric_name):
  METRIC2FUNC = {
    'Acc': accuracy_score,
    'MAE_Macro': mae_macro,
    'F1_Macro': f1_macro,
    'Recall_Macro': recall_macro
  }

  if metric_name in METRIC2FUNC:
    return METRIC2FUNC[metric_name]
  else:
    raise NotImplementedError('Metric %s is not implemented!' % metric_name)

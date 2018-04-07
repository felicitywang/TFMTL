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


def accuracy_score(y_trues, y_preds, labels):
  return sklearn.metrics.accuracy_score(y_true=y_trues,
                                        y_pred=y_preds,
                                        normalize=True  # return fraction
                                        )


def accurate_number(y_trues, y_preds):
  return sklearn.metrics.accuracy_score(y_true=y_trues,
                                        y_pred=y_preds,
                                        normalize=False  # return number
                                        )


def f1_macro(y_trues, y_preds, labels):
  """
  macro-averaged(unweighted mean) f1 score of all classes

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


def mae_macro(y_trues, y_preds):
  """
  macro-averaged(unweighted mean) mean absolute error

  :param y_trues: list of ground truth labels
  :param y_preds: list of predicted labels
  :return: float
  """
  return sklearn.metrics.mean_absolute_error(y_true=y_trues,
                                             y_pred=y_preds,
                                             sample_weight=None,
                                             multioutput='uniform_average')


def recall_macro(y_trues, y_preds, labels):
  """
  macro-averaged(unweighted mean) recall score of all classes

  :param y_trues: list of ground truth labels
  :param y_preds: list of predicted labels
  :param labels: labels for each class in a list, must specify
  :return: float
  """
  return sklearn.metrics.recall_score(y_true=y_trues,
                                      y_pred=y_preds,
                                      labels=labels,
                                      average='macro')


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

  if __name__ == '__main__':
    y_true = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
  y_pred = [0, 1, 1, 2, 2, 3, 3, 4, 4, 0]
  labels = [0, 1, 2, 3, 4]

  print(accuracy_score(y_trues=y_true, y_preds=y_pred))
  print(accurate_number(y_trues=y_true, y_preds=y_pred))

  print(mae_macro(y_trues=y_true, y_preds=y_pred))

  print(recall_macro(y_trues=y_true, y_preds=y_pred, labels=labels))
  print(f1_macro(y_trues=y_true, y_preds=y_pred, labels=labels))

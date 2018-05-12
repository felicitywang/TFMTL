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
# =============================================================================

"""Write TFRecord files for the given target task in the NAACL 18 paper

run: python write.py TARGET
where TARGET is either ALL or one of the Ruder tasks
"""
import os
import shutil
import sys

from expts.scripts import write_tfrecords_merged
from expts.scripts import write_tfrecords_single

# all the tasks in the NACCL paper(first author not Ruder, but to keep consitent with ruder_tokenizer and the experiment name RUDER_NACCL_18)
RUDER_TASKS = ['Topic2', 'Topic5', 'Target',
               'Stance', 'ABSA-L', 'ABSA-R', 'FNC-1', 'MultiNLI']

# NAACL paper table 4
RUDER_AUX_TASK_DICT = {
  'Topic2': ['FNC-1', 'MultiNLI', 'Target'],
  'Topic5': ['FNC-1', 'MultiNLI', 'ABSA-L', 'Target'],
  'Target': ['FNC-1', 'MultiNLI', 'Topic-5'],
  'Stance': ['FNC-1', 'MultiNLI', 'Target'],
  'ABSA-L': ['Topic5'],
  'ABSA-R': ['Topic5', 'ABSA-L', 'Target'],
  'FNC-1': ['Stance', 'MultiNLI', 'Topic5', 'ABSA-R', 'Target'],
  'MultiNLI': ['Topic5']
}

# argument files to use


ARGS_FILES = {
  'args_all_0.json': 'all_0/',  # all splits, min_freq = 0
  'args_train_1.json': 'train_1/'  # train split, min_freq = 1
}

if __name__ == '__main__':

  target_task = sys.argv[1]
  assert target_task in [
    'ALL'] + RUDER_TASKS, 'Target dataset %s not supported in this ' \
                          'experiment!' % target_task

  # single dataset
  # print('Writing TFRecord files for the single task %s with train split '
  #       'and min_freq = 1...' % target_task)

  for args_file in ARGS_FILES:
    folder_st_old = write_tfrecords_single.main(['', target_task, args_file])
    folder_st = os.path.join('data/tf/', target_task + '-st', ARGS_FILES[
      args_file])
    # print('old:', folder_st_old)
    # print('new:', folder_st)
    shutil.move(folder_st_old, folder_st)

  # multiple datasets
  if target_task == 'ALL':
    datasets = RUDER_TASKS
  else:
    datasets = [target_task] + RUDER_AUX_TASK_DICT[target_task]

  for args_file in ARGS_FILES:
    folder_mt_old = write_tfrecords_merged.main(
      [''] + datasets + [args_file])
    folder_mt = os.path.join('data/tf/', target_task + '-mt', ARGS_FILES[
      args_file])
    # print('old:', folder_mt_old)
    # print('new:', folder_mt)
    shutil.move(folder_mt_old, folder_mt)

  shutil.rmtree('data/tf/single/')
  shutil.rmtree('data/tf/merged')

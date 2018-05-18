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
TFRecord files would be saved in
data/tf/TARGET-st/{all_0, train_1}/TARGET/
data/tf/TARGET-mt/{all_0, train_1}/DATASET/, DATASET in [TARGET, AUXs]
"""
import os
import shutil
import sys
from pathlib import Path
from time import sleep

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
  'args_train_1.json': 'train_1/',  # train split, min_freq = 1
  # 'args_all_0_glove_init.json': 'all_0_glove_init',
  # 'args_train_1_glove_init.json': 'train_1_glove_init',
  # 'args_all_0_glove_expand.json': 'all_0_glove_expand',
  # 'args_train_1_glove_expand.json': 'train_1_glove_expand',
}

if __name__ == '__main__':

  target_task = sys.argv[1]
  assert target_task in [
    'ALL'] + RUDER_TASKS, 'Target dataset %s not supported in this ' \
                          'experiment!' % target_task

  if target_task != 'ALL':

    # single dataset
    for args_file in ARGS_FILES:
      folder_st = os.path.join('data/tf/', target_task + '-st', ARGS_FILES[
        args_file], target_task)
      if Path(folder_st).exists():
        print("File %s already exists! Skipping..." % folder_st)
        break
      else:
        print("File %s doesn't exits. Creating..." % folder_st)
      folder_st_old = write_tfrecords_single.main(['', target_task, args_file])
      shutil.move(folder_st_old, folder_st)
      os.symlink(os.path.abspath(folder_st), os.path.abspath(folder_st_old))

    # multiple datasets
    for args_file in ARGS_FILES:
      folder_mt = os.path.join('data/tf/', target_task + '-mt', ARGS_FILES[
        args_file])
      if Path(folder_mt).exists():
        print("File %s already exists! Skipping..." % folder_mt)
        break
      else:
        print("File %s doesn't exits. Creating..." % folder_mt)
      folder_mt_old = write_tfrecords_merged.main(
        ['', target_task] + RUDER_AUX_TASK_DICT[target_task] + [args_file])
      shutil.move(folder_mt_old, folder_mt)
      os.symlink(os.path.abspath(folder_mt), os.path.abspath(folder_mt_old))

  else:
    for args_file in ARGS_FILES:
      folder_mt = os.path.join('data/tf/', 'ALL-mt', ARGS_FILES[
        args_file])
      if Path(folder_mt).exists():
        print("File %s already exists! Skipping..." % folder_mt)
        break
      else:
        print("File %s doesn't exits. Creating..." % folder_mt)
      folder_mt_old = write_tfrecords_merged.main(
        [''] + RUDER_TASKS + [args_file])
      shutil.move(folder_mt_old, folder_mt)
      os.symlink(os.path.abspath(folder_mt), os.path.abspath(folder_mt_old))

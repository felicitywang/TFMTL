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
# See the License for the specific lang governing permissions and
# limitations under the License.
# =============================================================================

"""Read the tsv Turk data and convert to json format

Instead of binary labels, scores ranging in [0, 1] would be kept
"""

import gzip
import json
import os

import pandas as pd

from mtl.util.util import make_dir


def main():
  raw_dir = 'data/raw/TURK'
  json_dir = 'data/json/'

  domains = ['GOV', 'LIF', 'BUS', 'LAW', 'HEA', 'MIL', 'SPO']

  for domain in domains:
    tsvpath = os.path.join(raw_dir, domain + '.tsv')
    df = pd.read_csv(tsvpath, sep='\t')
    data = []
    index = 0
    for item in df.to_dict('records'):
      data.append({
        'index': index,
        'id': item['id'],
        'text': item['sent'],
        'label': float(item['score_mean']) / 100.0,
      })
      index += 1

    directory = os.path.join(json_dir, domain + '_turk_reg')
    make_dir(directory)
    with gzip.open(os.path.join(directory, 'data.json.gz'), mode='wt') as file:
      json.dump(data, file, ensure_ascii=False)

  # open test
  for domain in domains:
    directory = os.path.join(json_dir, domain + '_turk_reg')
    # print(directory)
    with gzip.open(os.path.join(directory, 'data.json.gz'), mode='rt') as file:
      test = json.load(file)
      print('{}: all={}'.format(
        directory,
        len(test)))


if __name__ == '__main__':
  main()

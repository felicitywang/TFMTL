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


"""Read the gold labels and translations for each domain, convert to json format


labels in:
data/raw/gold/labels/DOMAIN/1{A,B}/{speech,text}/....labels
translations in:
data/raw/gold/translations/{oracle, one, bop}/DOMAIN/1{A,B}/{speech,
text}/....labels

e.g.:
data/raw/gold/GOV/1A/domain_GOV.list (e.g. lines of file path like MATERIAL_BASE-1A_25144823)
the text would be in
1A/speech/MATERIAL_BASE-1A_25144823.txt or
1A/text/MATERIAL_BASE-1A_25144823.txt

'1A: GOV, LIF, BUS, LAW'
'1B: GOV, LIF, HEA, MIL'
"""

import gzip
import json
import os
from itertools import product

from mtl.util.util import make_dir

BASE_DIR = 'data/raw/gold'
TRANSLATIONS = [
  'oracle',
  'one',
  'bop'
]

DOMAIN_6 = ['GOV', 'LIF', 'HEA', 'LAW', 'MIL', 'BUS']

LANG_DIRS = {
  '1A': ['GOV', 'LIF', 'BUS', 'LAW'],
  '1B': ['GOV', 'LIF', 'HEA', 'MIL']
}


def main():
  domain_dirs = {}
  for lang in LANG_DIRS:
    for domain in LANG_DIRS[lang]:
      if domain not in domain_dirs:
        domain_dirs[domain] = [lang]
      else:
        domain_dirs[domain].append(lang)

  pos_filepaths = {domain: [] for domain in DOMAIN_6}
  pos_nums = {domain: 0 for domain in DOMAIN_6}

  for translation in TRANSLATIONS:

    for lang, domain in product(LANG_DIRS, DOMAIN_6):
      text_dir = os.path.join(BASE_DIR, 'translations', translation,
                              lang)
      label_dir = os.path.join(BASE_DIR, 'labels', domain, lang)
      filename = os.path.join(label_dir, 'domain_' + domain + '.list')
      if not os.path.exists(filename):
        print('{} doesn\'st exist. Skipping...'.format(filename))
        continue

      num = 0
      filepaths = []
      with open(filename) as file:
        print(filename)
        for line in file.readlines():
          num += 1
          for t in ['speech', 'text']:
            filepath = os.path.join(
              text_dir, t, line.strip() + '.txt')
            if os.path.exists(filepath):
              filepaths.append(filepath)
            else:
              pass
              # print('Can\'t find {}'.format(filepath))
        pos_filepaths[domain].extend(filepaths)
        pos_nums[domain] += num

      # for key, item in pos_filepaths.items():
      #     print(key, len(item))

      for domain in pos_nums:
        assert pos_nums[domain] == len(pos_filepaths[domain]), \
          str(pos_nums[domain]) + ' ' + \
          str(len(pos_filepaths[domain]))

      # make data.json

    all_filepaths = get_all_filepaths(translation)
    # for key, item in all_filepaths.items():
    #     print(key, len(item))

    # print(all_filepaths)

    # get gold data: pos + neg, index + text
    data = {}
    for domain in DOMAIN_6:
      data[domain] = []
      for filepath in all_filepaths[domain]:
        if filepath in pos_filepaths[domain]:
          label = 1
        else:
          label = 0
        data[domain].append({
          'text': read_file(filepath),
          'label': label
        })

    for key, item in data.items():
      print(key, len(item))

    # save data
    for domain in data:
      path = os.path.join('data/json/', domain +
                          '_gold_' + translation, 'data.json.gz')
      make_dir(os.path.dirname(path))
      with gzip.open(path, mode='wt') as file:
        json.dump(data[domain], file, ensure_ascii=False)

    # open for test
    print('Opening written json for test...')
    for domain in data:
      path = os.path.join('data/json/', domain +
                          '_gold_' + translation, 'data.json.gz')
      with gzip.open(path, mode='rt') as file:
        test = json.load(file)
        print('{}: pos={} neg={} all={}'.format(
          domain,
          len([i for i in test if int(i['label']) == 1]),
          len([i for i in test if int(i['label']) == 0]),
          len(test)))


def read_file(filepath):
  """Read lines of a file, convert to single line"""
  str = ""
  with open(filepath) as file:
    for line in file.readlines():
      str += line
  return str


def get_all_filepaths(translation):
  all_filepaths = {domain: [] for domain in DOMAIN_6}
  for lang, t in product(LANG_DIRS, ['speech', 'text']):
    directory = os.path.join(
      BASE_DIR, 'translations', translation, lang, t)
    for filepath in os.listdir(directory):
      if filepath.endswith('.txt'):
        for domain in LANG_DIRS[lang]:
          all_filepaths[domain].append(
            os.path.join(directory, filepath))
  return all_filepaths


if __name__ == '__main__':
  main()

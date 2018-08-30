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
# ==============================================================================

"""Convert plain text file to gzipped json file, used for predict mode"""

# -*- coding: utf-8 -*-

import gzip
import json
# path that saves
import sys

from tqdm import tqdm


"""Converts to data to predict in plain text format to JSON format

Usage: python convert_TEXT_to_JSON.py text_file_path json_file_path
"""

text_file_path = sys.argv[1]
json_file_path = sys.argv[2]

data = []

with open(text_file_path, 'r') as file:
  for index, line in tqdm(enumerate(file.readlines())):
    data.append({
      'text': line.strip(),
      'id': str(index)
    })

with gzip.open(json_file_path, mode='wt') as file:
  json.dump(data, file, ensure_ascii=False)
  file.close()

# open test
with gzip.open(json_file_path, mode='rt') as file:
  data = json.load(file, encoding='utf-8')
  print(len(data))

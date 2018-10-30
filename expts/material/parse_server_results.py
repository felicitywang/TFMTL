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

"""Parse the dev results from the server in csv format"""
import os
import sys

from material_constants import *


def main():
  dir = sys.argv[1]

  domains = ['BUS', 'GOV-A', 'GOV-B', 'HEA', 'LAW', 'LIF-A', 'LIF-B', 'MIL',
             'SPO-A', 'SPO-B']

  p_misses = dict()
  p_fas = dict()

  for filename in os.listdir(dir):
    if not filename.endswith('.csv'):
      continue
    with open(os.path.join(dir, filename)) as file:
      lines = [line.strip() for line in file.readlines() if
               line.strip()]
      # print(len(lines))
      # print(lines)
      for i in range(int(len(lines))):
        # score_names = lines[i * 3 + 1].split(',')

        # assert score_names == [
        #    'P_truePositive', 'P_miss', 'P_falseAlarm', 'P_trueNegative']

        scores = lines[i].split(',')
        domain_name = scores[0]
        scores = scores[1:]
        if domain_name in ['GOV', 'LIF', 'SPO']:
          if '1A' in filename:
            domain_name += '-A'
          elif '1B' in filename:
            domain_name += '-B'
          else:
            raise ValueError('wrong name ' + filename)

        # print(domain_name)
        # print(scores)

        p_misses[domain_name] = scores[1]
        p_fas[domain_name] = scores[2]

  # print(p_misses)
  # print(p_fas)

  print(' '.join(domains))

  print('p_miss')
  for domain_name in domains:
    print(p_misses[domain_name], end=',')

  print()
  print('p_fa')
  for domain_name in domains:
    print(p_fas[domain_name], end=',')

  print()


if __name__ == '__main__':
  main()

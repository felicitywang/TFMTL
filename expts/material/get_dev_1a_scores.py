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


"""Compute p_miss and p_fa locally for 1A DEV"""
import os
import sys

DOMAINS_1A = [
  'BUS',
  'GOV',
  'LAW',
  'LIF',
  'SPO'
]


def get_gold_doc_ids(filename):
  """return a dictionary of all the gold doc ids for each domain in DEV 1A"""
  gold_doc_ids = dict()
  all_files = []
  with open(filename) as file:
    for line in file.readlines()[1:]:
      line = line.strip().split()
      # print(line)
      if line[1] not in gold_doc_ids:
        gold_doc_ids[line[1]] = []
      gold_doc_ids[line[1]].append(line[0])
      all_files.append(line[0])
    # print(annotations)
    # for k, v in gold_doc_ids.items():
    #   gold_doc_ids[k] = set(v)
    #   print(k, len(v))

    # print(len(all_files), len(set(all_files)))
  return gold_doc_ids


def compute_Pmiss_Pfa(ref_docs, search_docs, N_total):
  '''Given two sets of documents, return Probability of Miss and False Alarm
  '''

  N_miss = len(ref_docs - search_docs)
  N_FA = len(search_docs - ref_docs)
  N_relevant = len(ref_docs)

  if N_relevant > 0:
    P_miss = N_miss * 1.0 / N_relevant
  else:
    # P_miss = 0.0
    P_miss = float('nan')
    # P_miss = -1.1

  P_FA = N_FA * 1.0 / (N_total - N_relevant)
  return P_miss, P_FA


def get_pred_doc_ids(folder):
  """get predicted doc ids saved in folder"""
  pred_doc_ids = dict()
  for domain in DOMAINS_1A:
    pred_doc_ids[domain] = []
    path = os.path.join(folder, domain + '.tsv')
    with open(path) as file:
      for line in file.readlines():
        [doc_id, label, score] = line.strip().split()
        if label == 'Y':
          pred_doc_ids[domain].append(doc_id)
        else:
          assert label == 'N', label
  # for k, v in pred_doc_ids.items():
  #   print(k, len(v))

  return pred_doc_ids


def get_all_doc_ids():
  all_doc_ids = []
  with open('dev_1a_all_doc_ids.txt') as file:
    for line in file.readlines():
      all_doc_ids.append(line.strip())
  return set(all_doc_ids)


def main():
  # all doc number for DEV
  # 1a text 217
  # 1a speech 449
  # 1b text 244
  # 1b speech speech 160

  folder = sys.argv[1]  # folder that contains d-domain.tgz for each domain
  # for 1A
  gold_doc_ids = get_gold_doc_ids('dev_1a_gold_doc_ids.tsv')
  pred_doc_ids = get_pred_doc_ids(folder)
  # num_all_doc_ids_dev_1a = len(get_all_doc_ids())
  num_all_doc_ids_dev_1a = 666

  for domain in DOMAINS_1A:
    print(domain, 'gold', len(gold_doc_ids[domain]),
          'pred', len(pred_doc_ids[domain]))

    p_miss, p_fa = compute_Pmiss_Pfa(
      set(gold_doc_ids[domain]), set(pred_doc_ids[domain]),
      num_all_doc_ids_dev_1a)

    print(p_miss, p_fa)


if __name__ == '__main__':
  main()

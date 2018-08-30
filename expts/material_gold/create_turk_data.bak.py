"""Create json data"""
import gzip
import json
import os

import numpy as np

from mtl.util.util import make_dir

"""
/data/json/raw/
train: synthetic 
dev:   half of gold 
test:  half of gold
"""

# TODO combine with onebest and bop

def parse_args():
  import argparse
  p = argparse.ArgumentParser()
  p.add_argument('--seed', type=int, default=42,
                 help='random seed')
  return p.parse_args()


def main():
  global seed
  args = parse_args()
  seed = args.seed

  DOMAINS = [
    'GOV',
    'LIF',
    'HEA',
    # 'LAW',
    'MIL',
    # 'BUS',
    # 'SPO'
  ]

  for domain in DOMAINS:
    make_dir(os.path.join('data/json/', domain + '_50.0'))

  for domain in DOMAINS:
    syn_path = os.path.join('data/json/', 'pilot_1_synthetic', domain +
                            '_50.0', 'data.json.gz')
    # gold_path = os.path.join('data/json/', 'pilot_2_gold_1B', domain +
    #                          '_50.0', 'data.json.gz')
    gold_path = os.path.join('data/json/', domain + 'g', 'data.json.gz')
    with gzip.open(gold_path, 'rt') as gold_file:
      gold_data = json.load(gold_file)
    with gzip.open(syn_path, 'rt') as syn_file:
      syn_data = json.load(syn_file)
    print(domain, len(gold_data), len(syn_data))

    # gold data
    get_gold_data(gold_data)

    data, index_dict = combine_data(gold_data, syn_data)
    dir = os.path.join('data/json/', domain + '_50.0')
    with gzip.open(os.path.join(dir, 'data.json.gz'), mode='wt') as file:
      json.dump(data, file, ensure_ascii=False)
    with gzip.open(os.path.join(dir, 'index.json.gz'), mode='wt') as file:
      json.dump(index_dict, file, ensure_ascii=False)

  # open test
  for domain in DOMAINS:
    dir = os.path.join('data/json/', domain + '_50.0')
    with gzip.open(os.path.join(dir, 'data.json.gz'), mode='rt') as file:
      data = json.load(file, encoding='utf-8')
    with gzip.open(os.path.join(dir, 'index.json.gz'), mode='rt') as file:
      index_dict = json.load(file, encoding='utf-8')
    print(domain,
          len(data),
          len(index_dict['train']),
          len(index_dict['valid']),
          len(index_dict['test']))


def get_gold_data(data):
  """Take half pos and half neg as dev/test, return index list"""
  global seed
  np.random.seed(seed)

  pos_list = []
  neg_list = []
  for i, d in enumerate(data):
    if int(d['label']) == 0:
      neg_list.append(i)
    else:
      pos_list.append(i)

  pos = np.random.permutation(np.array(pos_list))
  dev_pos, test_pos = map(list, np.split(pos, [int(len(pos_list) / 2)]))

  neg = np.random.permutation(np.array(neg_list))
  dev_neg, test_neg = map(list, np.split(neg, [int(len(neg_list) / 2)]))

  dev_pos.extend(dev_neg)
  test_pos.extend(test_neg)

  return dev_pos, test_pos


def combine_data(gold_data, syn_data):
  """Combine synthetic data and gold data, get new index_dict"""
  train_index = np.array(list(range(len(syn_data))))
  dev_index, test_index = get_gold_data(gold_data)

  data = []
  data.extend([syn_data[i] for i in train_index])
  data.extend([gold_data[i] for i in dev_index])
  data.extend([gold_data[i] for i in test_index])
  for index, item in enumerate(data):
    item['index'] = index
  index_dict = {
    'train': list(range(len(train_index))),
    'valid': list(range(len(train_index), len(train_index) + len(dev_index))),
    'test': list(range(len(train_index) + len(dev_index),
                       len(train_index) + len(dev_index) + len(test_index)))
  }

  return data, index_dict


if __name__ == '__main__':
  main()

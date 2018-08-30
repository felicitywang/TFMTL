"""Read the tsv MT data and convert to json format"""
import gzip
import json
import os
import shutil

import numpy as np

from mtl.util.util import make_dir

seed = 42


def main():
  pos_cuts = [90, 80, 70, 60, 50]
  neg_cuts = [50, 50, 50, 50, 50]

  json_dir = 'data/json/'

  domains = ['GOV', 'LIF', 'BUS', 'LAW', 'HEA', 'MIL', 'SPO']

  # combine with gold
  for pos, neg in zip(pos_cuts, neg_cuts):
    for domain in domains:
      gold_path = os.path.join('data/json/', domain + 'g', 'data.json.gz')
      if not os.path.exists(gold_path):
        print(gold_path)
        gold_data = []
      else:
        with gzip.open(gold_path, 'rt') as gold_file:
          gold_data = json.load(gold_file)
      dir = os.path.join(json_dir,
                         'TURK_' + domain + '_' + str(pos) + '_' + str(neg))
      syn_path = os.path.join(dir, 'data.json.gz')
      with gzip.open(syn_path, 'rt') as syn_file:
        syn_data = json.load(syn_file)
      print(domain, len(gold_data), len(syn_data))

      # gold data
      get_gold_data(gold_data)

      data, index_dict = combine_data(gold_data, syn_data)
      dir = os.path.join(json_dir,
                         'TURK_' + domain + '_' + str(pos) + '_' + str(
                           neg) + '_ORACLE')
      print(dir)
      make_dir(dir)
      with gzip.open(os.path.join(dir, 'data.json.gz'), mode='wt') as file:
        json.dump(data, file, ensure_ascii=False)
      print(os.path.exists(os.path.join(dir, 'data.json.gz')))
      with gzip.open(os.path.join(dir, 'index.json.gz'), mode='wt') as file:
        json.dump(index_dict, file, ensure_ascii=False)

  # open test
  for pos, neg in zip(pos_cuts, neg_cuts):
    for domain in domains:
      dir = os.path.join(json_dir,
                         'TURK_' + domain + '_' + str(pos) + '_' + str(
                           neg) + '_ORACLE')
      with gzip.open(os.path.join(dir, 'data.json.gz'), mode='rt') as file:
        data = json.load(file, encoding='utf-8')
      with gzip.open(os.path.join(dir, 'index.json.gz'), mode='rt') as file:
        index_dict = json.load(file, encoding='utf-8')
      print(domain,
            len(data),
            len(index_dict['train']),
            len(index_dict['valid']),
            len(index_dict['test']))
      if 'SPO' in dir:
        os.remove(os.path.join(dir, 'index.json.gz'))
        print(dir)
        print(os.listdir(dir))
      else:
        print(os.listdir(dir))
        assert 'index.json.gz' in (os.listdir(dir))

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
    'valid': list(
      range(len(train_index), len(train_index) + len(dev_index))),
    'test': list(range(len(train_index) + len(dev_index),
                       len(train_index) + len(dev_index) + len(test_index)))
  }

  return data, index_dict


if __name__ == '__main__':
  main()

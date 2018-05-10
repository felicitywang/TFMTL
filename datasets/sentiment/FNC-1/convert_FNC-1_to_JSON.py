'''Transform original Stance file into json format using pytreebank parser'''

# -*- coding: utf-8 -*-

import csv
import gzip
import json
import os
import sys

from tqdm import tqdm

FNC_LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

LABEL_DICT = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}


def readFNCData(datafolder="./data/", debug=True,
                num_instances=20):
  data_train = {"seq1": [], "seq2": [], "stance": [], "labels": []}
  data_train = parseFNC(datafolder,
                        "fakenewschallenge/train_bodies.csv",
                        "fakenewschallenge/trainsplit_stances.csv",
                        data_train, debug, num_instances)
  data_dev = {"seq1": [], "seq2": [], "stance": [], "labels": []}
  data_dev = parseFNC(datafolder,
                      "fakenewschallenge/train_bodies.csv",
                      "fakenewschallenge/devsplit_stances.csv",
                      data_dev, debug, num_instances)
  data_test = {"seq1": [], "seq2": [], "stance": [], "labels": []}
  data_test = parseFNC(datafolder,
                       "fakenewschallenge/competition_test_bodies.csv",
                       "fakenewschallenge/competition_test_stances.csv",
                       data_test, debug, num_instances)
  data_train["labels"] = sorted(data_train["labels"])
  assert data_train["labels"] == FNC_LABELS
  data_dev["labels"] = data_train["labels"]
  data_test["labels"] = data_train["labels"]
  return data_train, data_dev, data_test


def parseFNC(datafolder, datafile_bodies, datafile_stances,
             data_dict, debug, num_instances):
  id2body = {}
  with open(os.path.join(datafolder, datafile_bodies), 'r',
            encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    i = -1
    for row in csvreader:
      i += 1
      if i == 0:
        continue
      body_id, body = row
      id2body[body_id] = body

  with open(os.path.join(datafolder, datafile_stances), 'r',
            encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    i = -1
    for row in csvreader:
      i += 1
      if i == 0:
        continue
      if debug and i >= num_instances + 1:
        continue
      headline, body_id, stance = row
      data_dict["seq1"].append(headline)
      data_dict["seq2"].append(id2body[body_id])
      data_dict["stance"].append(stance)

  for lab in set(data_dict["stance"]):
    data_dict["labels"].append(lab)

  return data_dict


def make_example_list(d, starting_index):
  index = starting_index

  examples = list(zip(*[d['seq1'],
                        d['seq2'],
                        d['stance']
                        ]))

  res = []
  for example in tqdm(examples):
    ex = dict()
    ex['index'] = index
    ex['seq1'] = example[0]
    ex['seq2'] = example[1]
    ex['label'] = LABEL_DICT[example[2]]
    res.append(ex)
    index += 1

  ending_index = index  # next example will have this index
  index_list = list(range(starting_index, ending_index))
  return res, ending_index, index_list


import random
from csv import DictReader
from csv import DictWriter


# Define data class


class FNCData:
  """
  Define class for Fake News Challenge data
  """

  def __init__(self, file_instances, file_bodies):

    # Load data
    self.instances = self.read(file_instances)
    bodies = self.read(file_bodies)
    self.heads = {}
    self.bodies = {}

    # Process instances
    for instance in self.instances:
      if instance['Headline'] not in self.heads:
        head_id = len(self.heads)
        self.heads[instance['Headline']] = head_id
      instance['Body ID'] = int(instance['Body ID'])

    # Process bodies
    for body in bodies:
      self.bodies[int(body['Body ID'])] = body['articleBody']

  def read(self, filename):
    """
    Read Fake News Challenge data from CSV file
    Args:
        filename: str, filename + extension
    Returns:
        rows: list, of dict per instance
    """

    # Initialise
    rows = []

    # Process file
    with open(filename, "r", encoding='utf-8') as table:
      r = DictReader(table)
      for line in r:
        rows.append(line)

    return rows


def split_seen(data, rand=False, prop_dev=0.2, rnd_sd=1489215):
  """

  Split data into separate sets with overlapping headlines

  Args:
      data: FNCData object
      rand: bool, True: random split and False: use seed for official baseline split
      prop_dev: float, proportion of data for dev set
      rnd_sd: int, random seed to use for split

  Returns:
      train: list, of dict per instance
      dev: list, of dict per instance

  """

  # Initialise
  list_bodies = [body for body in data.bodies]
  n_dev_bodies = round(len(list_bodies) * prop_dev)
  r = random.Random()
  if rand is False:
    r.seed(rnd_sd)
  train = []
  dev = []

  # Generate list of bodies for dev set
  r.shuffle(list_bodies)
  list_dev_bodies = list_bodies[-n_dev_bodies:]

  # Generate train and dev sets
  for stance in data.instances:
    if stance['Body ID'] not in list_dev_bodies:
      train.append(stance)
    else:
      dev.append(stance)

  return train, dev


def split_unseen(data, rand=False, prop_dev=0.2, rnd_sd=1489215):
  """

  Split data into completely separate sets (i.e. non-overlap of headlines and bodies)

  Args:
      data: FNCData object
      rand: bool, True: random split and False: constant split
      prop_dev: float, target proportion of data for dev set
      rnd_sd: int, random seed to use for split

  Returns:
      train: list, of dict per instance
      dev: list, of dict per instance

  """

  # Initialise
  n = len(data.instances)
  n_dev = round(n * prop_dev)
  dev_ind = {}
  r = random.Random()
  if rand is False:
    r.seed(rnd_sd)
  train = []
  dev = []

  # Identify instances for dev set
  while len(dev_ind) < n_dev:
    rand_ind = r.randrange(n)
    if not data.instances[rand_ind]['Stance'] in ['agree', 'disagree',
                                                  'discuss']:
      continue
    if rand_ind not in dev_ind:
      rand_head = data.instances[rand_ind]['Headline']
      rand_body_id = data.instances[rand_ind]['Body ID']
      dev_ind[rand_ind] = 1
      track_heads = {}
      track_bodies = {}
      track_heads[rand_head] = 1
      track_bodies[rand_body_id] = 1
      pre_len_heads = len(track_heads)
      pre_len_bodies = len(track_bodies)
      post_len_heads = 0
      post_len_bodies = 0
      while pre_len_heads != post_len_heads and pre_len_bodies != post_len_bodies:
        pre_len_heads = len(track_heads)
        pre_len_bodies = len(track_bodies)
        for i, stance in enumerate(data.instances):
          if not data.instances[i]['Stance'] in ['agree', 'disagree',
                                                 'discuss']:
            continue
          if i != rand_ind and (stance['Headline'] in track_heads or stance[
            'Body ID'] in track_bodies):
            track_heads[stance['Headline']] = 1
            track_bodies[stance['Body ID']] = 1
        post_len_heads = len(track_heads)
        post_len_bodies = len(track_bodies)

      for k, stance in enumerate(data.instances):
        if k != rand_ind and (stance['Headline'] in track_heads or stance[
          'Body ID'] in track_bodies) and (
          stance['Stance'] in ['agree', 'disagree', 'discuss']):
          dev_ind[k] = 1

  # Generate train and dev sets
  for k, stance in enumerate(data.instances):
    if k in dev_ind:
      dev.append(stance)
    else:
      train.append(stance)

  return train, dev


def save_csv(data_split, filepath):
  """
  Save predictions to CSV file
  Args:
      pred: numpy array, of numeric predictions
      file: str, filename + extension
  """

  with open(filepath, 'w', encoding='utf-8') as csvfile:
    fieldnames = ['Headline', 'Body ID', 'Stance']
    writer = DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for instance in data_split:
      writer.writerow(
        {'Headline': instance["Headline"], 'Body ID': instance["Body ID"],
         'Stance': instance["Stance"]})


if __name__ == '__main__':
  datafolder = sys.argv[1]

  # split train to train + dev
  data = FNCData(os.path.join(datafolder,
                              "fakenewschallenge/train_stances.csv"),
                 os.path.join(datafolder,
                              "fakenewschallenge/train_bodies.csv"))
  train, dev = split_unseen(data)
  save_csv(train, os.path.join(datafolder,
                               "fakenewschallenge/trainsplit_stances.csv"))
  save_csv(dev, os.path.join(datafolder, "fakenewschallenge/devsplit_stances.csv"))

  data_train, data_dev, data_test = readFNCData(
    datafolder=datafolder,
    debug=False,
    num_instances=20)

  index = 0
  train_list, index, train_index = make_example_list(data_train, index)
  dev_list, index, dev_index = make_example_list(data_dev, index)
  test_list, index, test_index = make_example_list(data_test, index)

  index_dict = {
    'train': train_index,
    'valid': dev_index,
    'test': test_index
  }

  assert len(set(train_index).intersection(set(dev_index))) == 0
  assert len(set(train_index).intersection(set(test_index))) == 0
  assert len(set(dev_index).intersection(set(test_index))) == 0

  data_list = []
  data_list.extend(train_list)
  data_list.extend(dev_list)
  data_list.extend(test_list)

  # write out to JSON files
  #  index.json.gz
  with gzip.open(datafolder + 'index.json.gz', mode='wt') as file:
    json.dump(index_dict, file, ensure_ascii=False)
  #  data.json.gz
  with gzip.open(datafolder + 'data.json.gz', mode='wt') as file:
    json.dump(data_list, file, ensure_ascii=False)

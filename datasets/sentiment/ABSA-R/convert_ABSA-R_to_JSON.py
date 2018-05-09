import gzip
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

from sklearn.model_selection import train_test_split
from tqdm import tqdm

ABSA_LABELS = ['negative', 'neutral', 'positive']
LABEL_DICT = {'negative': 0, 'neutral': 1, 'positive': 2}


def read_absa(domain, datafolder="./data/", debug=True, num_instances=20):
  assert domain in ['restaurants',
                    'restaurants'], '%s is not a valid domain.' % domain
  absa_path = os.path.join(datafolder, 'semeval2016-task5-absa-english')
  train_path = os.path.join(absa_path, '%s_english_training.xml' % domain)
  test_path = os.path.join(absa_path, '%s_english_test.xml' % domain)
  for path_ in [absa_path, train_path, test_path]:
    assert os.path.exists(path_), 'Error: %s does not exist.' % path_

  data_train = parse_absa(train_path, debug, num_instances)
  data_test = parse_absa(test_path)

  # trial data is a subset of training data; instead we split the train data
  data_train, data_dev = split_train_data(data_train)
  return data_train, data_dev, data_test


def parse_absa(file_path, debug=False, num_instances=20):
  """
  Extracts all reviews from an XML file and returns them as a list of Review objects.
  Adds a NONE aspect to all sentences with no aspect.
  :param file_path: the path of the XML file
  :return: a list of Review objects each containing a list of Sentence objects and other attributes
  """
  data = {"seq1": [], "seq2": [], "stance": []}
  e = ET.parse(file_path).getroot()
  for i, review_e in enumerate(e):
    if debug and i >= num_instances + 1:
      continue
    for sentence_e in review_e.find('sentences'):
      text = sentence_e.find('text').text
      # we do not care about sentences that do not contain an aspect
      if sentence_e.find('Opinions') is not None:
        for op in sentence_e.find('Opinions'):
          # the category is of the form ENTITY#ATTRIBUTE, e.g. LAPTOP#GENERAL
          target = ' '.join(op.get('category').split('#'))
          polarity = op.get('polarity')
          data['seq1'].append(target)
          data['seq2'].append(text)
          data['stance'].append(polarity)
  data["labels"] = sorted(list(set(data["stance"])))
  assert data["labels"] == ABSA_LABELS
  return data


def split_train_data(data_train):
  """Split the train data into train and dev data."""
  train_ids, _ = train_test_split(range(len(data_train['seq1'])),
                                  test_size=0.1, random_state=42)
  data_dev = defaultdict(list)
  new_data_train = defaultdict(list)
  for key, examples in data_train.items():
    if key == 'labels':
      continue
    # no numpy indexing, so we iterate over the examples
    for i, example in enumerate(examples):
      if i in train_ids:
        new_data_train[key].append(example)
      else:
        data_dev[key].append(example)
  new_data_train['labels'] = data_train['labels']
  data_dev['labels'] = data_train['labels']
  return new_data_train, data_dev


def make_example_list(d, starting_index):
  index = starting_index

  seq1_list = d["seq1"]
  seq2_list = d["seq2"]
  stance_list = d["stance"]

  examples = list(zip(*[seq1_list, seq2_list, stance_list]))

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


if __name__ == "__main__":
  datafolder = sys.argv[1]

  data_train, data_dev, data_test = read_absa('restaurants',
                                              datafolder,
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

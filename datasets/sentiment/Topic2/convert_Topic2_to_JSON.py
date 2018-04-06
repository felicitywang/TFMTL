import os
import json
import gzip
import sys

from constants import TOPIC_LABELS


def readTopicBased(datafolder="./data/", debug=True, num_instances=20):
  topic_based_path = os.path.join(
    datafolder,
    'semeval2016-task4b-topic-based-sentiment')
  train_path = os.path.join(
    topic_based_path,
    '100_topics_XXX_tweets.topic-two-point.subtask-BD.train.gold_downloaded.tsv')  # NOQA
  dev1_path = os.path.join(
    topic_based_path,
    '100_topics_XXX_tweets.topic-two-point.subtask-BD.dev.gold_downloaded.tsv')
  dev2_path = os.path.join(
    topic_based_path,
    '100_topics_XXX_tweets.topic-two-point.subtask-BD.devtest.gold_downloaded.tsv')  # NOQA
  test_data_path = os.path.join(
    topic_based_path,
    'SemEval2016-task4-test.subtask-BD.txt')
  test_labels_path = os.path.join(
    topic_based_path,
    'SemEval2016_task4_subtaskB_test_gold.txt')

  for path_ in [topic_based_path,
                train_path,
                dev1_path,
                dev2_path,
                test_data_path,
                test_labels_path]:
    assert os.path.exists(path_), 'Error: %s does not exist.' % path_

  data_train = parse_topic_based(train_path, debug, num_instances)
  data_dev1 = parse_topic_based(dev1_path, debug, num_instances)
  data_dev2 = parse_topic_based(dev2_path, debug, num_instances)
  data_test = parse_topic_test_data(test_data_path, test_labels_path)
  assert data_train["labels"] == TOPIC_LABELS
  data_dev1["labels"] = data_train["labels"]
  data_test["labels"] = data_train["labels"]

  # add the second dev data to the train set
  data_train["seq1"] += data_dev2["seq1"]
  data_train["seq2"] += data_dev2["seq2"]
  data_train["stance"] += data_dev2["stance"]
  return data_train, data_dev1, data_test


def parse_topic_based(file_path, debug=False, num_instances=20):
  data = {"seq1": [], "seq2": [], "stance": []}
  with open(file_path) as f:
    for i, line in enumerate(f):
      id_, target, sentiment, tweet = line.split('\t')
      try:
        sentiment = float(sentiment)
      except ValueError:
        pass
      if debug and i >= num_instances+1:
        continue
      if tweet.strip() == 'Not Available':
        continue
      data["seq1"].append(target)
      data["seq2"].append(tweet)
      data["stance"].append(sentiment)

  # we have to sort the labels so that they're in the order
  # -2,-1,0,1,2 and are mapped to 0,1,2,3,4 (for subtask C)
  data["labels"] = sorted(list(set(data["stance"])))
  return data


def parse_topic_test_data(examples_path, labels_path):
  # Note: no debugging for the test data (20k tweets for subtask C)
  data = {"seq1": [], "seq2": [], "stance": []}
  with open(examples_path) as f_examples, open(labels_path) as f_labels:
    for i, (line_examples, line_labels) in enumerate(zip(f_examples,
                                                         f_labels)):
      _, examples_target, _, *tweet = line_examples.strip().split('\t')
      # two lines contain a tweet, for some reason
      _, labels_target, sentiment, *_ = line_labels.strip().split('\t')
      # one test tweet contains a tab character
      if isinstance(tweet, list):
        tweet = '\t'.join(tweet)
      try:
        sentiment = float(sentiment)
      except ValueError:
        pass

      s = "%s != %s at line %d in files %s and %s." % (examples_target,
                                                       labels_target,
                                                       i,
                                                       examples_path,
                                                       labels_path)
      assert examples_target == labels_target, s

      if tweet.strip() == 'Not Available':
        continue
      data["seq1"].append(examples_target)
      data["seq2"].append(tweet)
      data["stance"].append(sentiment)
  data["labels"] = sorted(list(set(data["stance"])))
  return data


def make_example_list(d, starting_index):
  index = starting_index

  seq1_list = d["seq1"]
  seq2_list = d["seq2"]
  stance_list = d["stance"]

  examples = list(zip(*[seq1_list, seq2_list, stance_list]))

  res = []
  for example in examples:
    ex = dict()
    ex['index'] = index
    ex['seq1'] = example[0]
    ex['seq2'] = example[1]
    label = example[2]
    if label == "positive":
      label = 1
    elif label == "negative":
      label = 0
    else:
      raise ValueError("Unrecognized label: {}".format(label))
    ex['label'] = label
    res.append(ex)
    index += 1

  ending_index = index  # next example will have this index
  index_list = list(range(starting_index, ending_index))
  return res, ending_index, index_list


if __name__ == "__main__":
  try:
    datafolder = sys.argv[1]
  except:
    datafolder = "./TOPIC2/"

  data_train, data_dev1, data_test = readTopicBased(datafolder=datafolder,
                                                    debug=True,
                                                    num_instances=20)

  index = 0
  train_list, index, train_index = make_example_list(data_train, index)
  dev_list, index, dev_index = make_example_list(data_dev1, index)
  test_list, index, test_index = make_example_list(data_test, index)

  index_dict = {
    'train': train_index,
    'dev': dev_index,
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

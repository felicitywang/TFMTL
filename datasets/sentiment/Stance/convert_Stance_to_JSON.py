'''Transform original Stance file into json format using pytreebank parser'''

# -*- coding: utf-8 -*-

import csv
import gzip
import json
import os
import sys

from tqdm import tqdm

LABEL_DICT = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}
STANCE_LABELS = ['AGAINST', 'FAVOR', 'NONE']
datafolder = sys.argv[1]


def parse_semeval_csv(filepath, empty_dict_1, empty_dict_2, mode, debug=False,
                      num_instances=20):
    with open(filepath, 'r', encoding='latin-1') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        i = -1
        for row in tqdm(csvreader):
            i += 1
            if i == 0:
                continue
            if debug and i >= num_instances + 1:
                continue
            tweet, target, stance, opinion_towards, sentiment = row
            dict_chosen = empty_dict_1
            if target == 'Hillary Clinton':
                dict_chosen = empty_dict_2
            if mode == 'train' or target == 'Hillary Clinton' or (
                mode == 'test' and target == 'Donald Trump'):
                dict_chosen['seq1'].append(target)
                dict_chosen['seq2'].append(tweet)
                dict_chosen['stance'].append(stance)
                dict_chosen['opinion_towards'].append(opinion_towards)
                dict_chosen['sentiment'].append(sentiment)
    return empty_dict_1, empty_dict_2


def readSemEval2016Task6(datafolder='./', debug=True, num_instances=20):
    data_train = {'seq1': [], 'seq2': [], 'stance': [],
                  'opinion_towards': [], 'sentiment': [], 'labels': []}
    data_dev = {'seq1': [], 'seq2': [], 'stance': [],
                'opinion_towards': [], 'sentiment': [], 'labels': []}
    data_test = {'seq1': [], 'seq2': [], 'stance': [],
                 'opinion_towards': [], 'sentiment': [], 'labels': []}
    data_train, data_dev = parse_semeval_csv(os.path.join(
        datafolder, 'StanceDataset/train.csv'), data_train, data_dev,
        'train', debug, num_instances)
    data_test, data_dev = parse_semeval_csv(
        os.path.join(datafolder, 'StanceDataset/test.csv'),
        data_test, data_dev, 'test', False,
        num_instances)  # setting debug to False to get all test instances

    # For the final task training, the dev set is used as part of the training set
    for i, inst in tqdm(enumerate(data_dev['stance'])):
        data_train['seq1'].append(data_dev['seq1'][i])
        data_train['seq2'].append(data_dev['seq2'][i])
        data_train['stance'].append(data_dev['stance'][i])
        data_train['opinion_towards'].append(data_dev['opinion_towards'][i])
        data_train['sentiment'].append(data_dev['sentiment'][i])

    # sort the labels so that they are always in the same order so that we can
    # compute averaged positive and negative F1 (AGAINST, FAVOR, NONE)
    labels = sorted(list(set(data_train['stance'])))
    assert labels == STANCE_LABELS
    data_train['labels'] = labels
    data_dev['labels'] = labels
    data_test['labels'] = labels

    # we do not use the raw data ATM to correspond with the signature of the other data readers
    return data_train, data_dev, data_test


def make_example_list(d, starting_index):
    index = starting_index

    examples = list(zip(*[d['seq1'],
                          d['seq2'],
                          d['sentiment'],
                          d['stance'],
                          # d['opinion_towards']
                          ]))

    res = []
    for example in tqdm(examples):
        ex = dict()
        ex['index'] = index
        ex['seq1'] = example[0]
        ex['seq2'] = example[1]
        ex['sentiment'] = example[2]
        ex['label'] = LABEL_DICT[example[3]]
        res.append(ex)
        index += 1

    ending_index = index  # next example will have this index
    index_list = list(range(starting_index, ending_index))
    return res, ending_index, index_list


if __name__ == '__main__':
    datafolder = sys.argv[1]

    data_train, data_dev, data_test = readSemEval2016Task6(
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

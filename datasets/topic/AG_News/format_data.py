import pandas as pd
import json
import gzip
import numpy as np


file = open('train.csv', 'r')
train_list = []
for line in file.readlines():
    line = line.split(',')
    train_list.append({
        'label': line[0][1:-1],
        'title': line[1][1:-1],
        'description': line[2][1:-1]
    })


file = open('test.csv', 'r')
test_list = []
for line in file.readlines():
    line = line.split(',')
    test_list.append({
        'label': line[0][1:-1],
        'title': line[1][1:-1],
        'description': line[2][1:-1]
    })

# indices
train_index = list(range(len(train_list)))
test_index = list(range(len(train_list), len(train_list) + len(test_list)))
index = {
    'train': train_index,
    'test': test_index
}
assert len(set(index['train']).intersection(index['test'])) == 0

with gzip.open('index.json.gz', mode='wt') as file:
    json.dump(index, file)


all_list = train_list
all_list.extend(test_list)

with gzip.open('data.json.gz', mode='wt') as file:
    json.dump(all_list, file)


# test
with gzip.open('data.json.gz', mode='rt') as file:
    data_list = json.load(file)

with gzip.open('index.json.gz', 'rt') as file:
    index_dict = json.load(file)
    assert len(set(index['train']).intersection(index['test'])) == 0

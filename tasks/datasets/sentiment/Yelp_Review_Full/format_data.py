#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import json

file = open('train.csv', 'r')
train_list = []
for line in file.readlines():
    line = line.split(',')
    train_list.append({
        'label': line[0][1:-1],
        'text': line[1][1:-1]
    })
file = open('train.json', 'w')
json.dump(train_list, file)


file = open('test.csv', 'r')
test_list = []
for line in file.readlines():
    line = line.split(',')
    test_list.append({
        'label': line[0][1:-1],
        'text': line[1][1:-1]
    })
file = open('test.json', 'w')
json.dump(test_list, file)

train_list.extend(test_list)
file = open('data.json', 'w')
json.dump(train_list, file)

file = open('data.json', 'r')
data = json.load(file)

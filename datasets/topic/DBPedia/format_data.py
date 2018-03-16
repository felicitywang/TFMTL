#!/usr/bin/env python
# -*- coding=utf-8 -*-

import json

file = open('train.csv', 'r', encoding='utf-8')
train_list = []
for line in file.readlines():
    line = line.split(',')
    train_list.append({
        'label': line[0],
        'title': line[1][1:-1],
        'content': line[2][1: -1]
    })
file = open('train.json', 'w', encoding='utf-8')
json.dump(train_list, file)

file = open('test.csv', 'r', encoding='utf-8')
test_list = []
for line in file.readlines():
    line = line.split(',')
    test_list.append({
        'label': line[0],
        'title': line[1][1:-1],
        'content': line[2][1: -1]
    })
file = open('test.json', 'w', encoding='utf-8')
json.dump(test_list, file)

train_list.extend(test_list)
file = open('data.json', 'w', encoding='utf-8')
json.dump(train_list, file)

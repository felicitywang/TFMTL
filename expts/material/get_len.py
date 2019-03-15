import gzip
import json
import os
import sys

name = sys.argv[1]
with gzip.open(os.path.join('data/json', name, 'data.json.gz'),
               mode='rt') as file:
    data = json.load(file, encoding='utf-8')
    pos = [i for i in data if str(i['label']) == '1']
    neg = [i for i in data if str(i['label']) == '0']
    print(len(data), len(pos), len(neg))

if os.path.exists(os.path.join('data/json', name, 'index.json.gz')):
    with gzip.open(os.path.join('data/json', name, 'index.json.gz'),
                   mode='rt') as file:
        indices = json.load(file)
        train_data = [data[i] for i in indices['train']]
        valid_data = [data[i] for i in indices['valid']]

        pos = [i for i in train_data if str(i['label']) == '1']
        neg = [i for i in train_data if str(i['label']) == '0']
        print(len(train_data), len(pos), len(neg))

        pos = [i for i in valid_data if str(i['label']) == '1']
        neg = [i for i in valid_data if str(i['label']) == '0']
        print(len(valid_data), len(pos), len(neg))

text = [len(i['text'].strip().split()) for i in data]
print(sum(text) / len(text))

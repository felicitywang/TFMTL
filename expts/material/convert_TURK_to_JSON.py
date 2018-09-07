"""Read the tsv MT data and convert to json format"""
import gzip
import json
import os

import numpy as np
import pandas as pd

from mtl.util.util import make_dir


def main():
    pos_cuts = [90, 80, 70, 60, 50]
    neg_cuts = [50, 50, 50, 50, 50]

    raw_dir = 'data/raw/TURK'
    json_dir = 'data/json/'

    domains = ['GOV', 'LIF', 'BUS', 'LAW', 'HEA', 'MIL', 'SPO']

    for pos, neg in zip(pos_cuts, neg_cuts):
        for domain in domains:
            tsvpath = os.path.join(raw_dir, domain + '.tsv')
            df = pd.read_csv(tsvpath, sep='\t')
            data = []
            index = 0
            for item in df.to_dict('records'):
                score = float(item['score_mean'])
                if score >= neg and score <= pos:
                    # print(score)
                    continue
                # print(score)
                if score < neg:
                    label = 0
                else:
                    assert score > pos
                    label = 1
                data.append({
                    'index': index,
                    'id': item['id'],
                    'text': item['sent'],
                    'score': score / 100.0,
                    'label': label
                })
                index += 1

            dir = os.path.join(json_dir,
                               'TURK_' + domain + '_' + str(pos) + '_' + str(neg))
            # print(dir)
            make_dir(dir)
            with gzip.open(os.path.join(dir, 'data.json.gz'), mode='wt') as file:
                json.dump(data, file, ensure_ascii=False)

    # open test
    for pos, neg in zip(pos_cuts, neg_cuts):
        for domain in domains:
            dir = os.path.join(json_dir,
                               'TURK_' + domain + '_' + str(pos) + '_' + str(neg))
            print(dir)
            with gzip.open(os.path.join(dir, 'data.json.gz'), mode='rt') as file:
                test = json.load(file)
                # labels = [i['label'] for i in test]
                # print(labels)
                test_pos = [i for i in test if int(i['label']) == 1]
                test_neg = [i for i in test if int(i['label']) == 0]
                print(len(test), len(test_pos), len(test_neg))


if __name__ == '__main__':
    main()

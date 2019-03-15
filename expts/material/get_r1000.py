import gzip
import json
import os

from mtl.util.util import make_dir


def main():
    domains = ['GOV', 'LIF', 'BUS', 'LAW', 'HEA', 'MIL', 'SPO']

    for domain in domains:

        path = os.path.join('data/json', domain +
                            '_syn_p1000r1000', 'data.json.gz')

        print(path)
        with gzip.open(path, mode='rt') as file:
            data = json.load(file, encoding='utf-8')
        pos_data = []
        neg_data = []
        for i in data:
            if int(i['label']) == 1:
                pos_data.append(i)
            else:
                assert int(i['label']) == 0
                neg_data.append(i)

        make_dir(os.path.join('data/json', domain + '_syn_p1000 '))
        with gzip.open(
            os.path.join('data/json', domain + '_syn_p1000 ', 'data.json.gz'),
            mode='wt') as file:
            json.dump(pos_data, file, ensure_ascii=False)

        make_dir(os.path.join('data/json', domain + '_syn_r1000'))
        with gzip.open(
            os.path.join('data/json', domain + '_syn_r1000', 'data.json.gz'),
            mode='wt') as file:
            json.dump(neg_data, file, ensure_ascii=False)


if __name__ == '__main__':
    main()

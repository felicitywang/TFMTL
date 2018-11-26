"""Split turk data"""
import gzip
import json
import os
import random

from mtl.util.util import make_dir

domains = [
    "GOV",
    "LIF",
    "BUS",
    "LAW",
    "HEA",
    "MIL",
    "SPO"
]


def main():
    turk_name = '_turk_60_50'
    for domain in domains:
        folder = os.path.join('data/json', domain + turk_name)

        with gzip.open(os.path.join(folder, 'data.json.gz'), mode='rt') as file:
            data = json.load(file, encoding='utf-8')

        random.shuffle(data)

        l = len(data) // 5
        for i in range(5):
            tmp = data[: i * l + l]
            print(len(tmp))
            folder = os.path.join('data/json', domain +
                                  turk_name + '_' + str(i + 1))
            make_dir(folder)
            with gzip.open(os.path.join(folder, 'data.json.gz'), mode='wt') as file:
                print(os.path.join(folder, 'data.json.gz'))
                json.dump(tmp, file, ensure_ascii=False)


if __name__ == '__main__':
    main()

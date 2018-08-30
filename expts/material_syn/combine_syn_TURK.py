"""Use syn data as train set and TURK as dev"""
import gzip
import json
import os

from mtl.util.util import make_dir


def main():
    domains = ['GOV', 'LIF', 'BUS', 'LAW', 'HEA', 'MIL', 'SPO']

    syn_suffix = '_syn_1000'
    TURK_prefix = 'TURK_'
    TURK_suffixes = [
        '_90_50',
        '_80_50',
        '_70_50',
        '_60_50',
        '_50_50'
    ]

    base_dir = 'data/json/'
    for TURK_suffix in TURK_suffixes:
        for domain in domains:
            syn_path = os.path.join(
                base_dir, domain + syn_suffix, 'data.json.gz')
            turk_path = os.path.join(base_dir, TURK_prefix + domain + TURK_suffix,
                                     'data.json.gz')
            with gzip.open(syn_path, mode='rt') as file:
                syn_data = json.load(file)

            with gzip.open(turk_path, mode='rt') as file:
                turk_data = json.load(file)

            index_dict = {
                'train': list(range(len(syn_data))),
                'valid': list(range(len(syn_data), len(syn_data) + len(
                    turk_data))),
                'test': []
            }

            data = syn_data + turk_data
            dout = os.path.join('data/json/', domain +
                                syn_suffix + '_TURK' + TURK_suffix)
            make_dir(dout)
            print(dout)
            with gzip.open(os.path.join(dout, 'data.json.gz'), mode='wt') as file:
                json.dump(data, file, ensure_ascii=False)
            with gzip.open(os.path.join(dout, 'index.json.gz'), mode='wt') as file:
                json.dump(index_dict, file, ensure_ascii=False)


if __name__ == '__main__':
    main()

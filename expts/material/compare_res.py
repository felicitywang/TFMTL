import json

subdirs = {
    "1A": [
        "t6/mt-4.asr-s5",
        "tt18"
    ],
    "1B": [
        "t6/mt-5.asr-s5",
        "tt20"
    ]
}
eval_dir = 'DEV'
text_type = 'doc'

import os

domains = ['GOV', 'LIF', 'BUS', 'LAW', 'HEA', 'MIL', 'SPO']

from itertools import product
import sys

turk = "_turk_50_50_min_1_max_-1_vocab_-1_doc_-1_tok_tweet_dan_meanmax_relu_0.1_nopretrain"
turk_reg = "_turk_reg_min_1_max_-1_vocab_-1_doc_-1_tok_tweet_dan_meanmax_relu_0.1_nopretrain"
root_dir = 'data/predictions/'


def main():
    all_res = {}

    dataset_suffixes = sys.argv[1]

    num = 0

    all_dirs = {}
    for lang in subdirs:

        directory = os.path.join(text_type, lang, eval_dir)

        if directory in all_dirs:
            all_dirs[directory].append(subdirs[lang])
        else:
            all_dirs[directory] = subdirs[lang]

    for basedir, domain, dataset_suffix in product(
        all_dirs, domains, dataset_suffixes):
        for subdir in all_dirs[basedir]:
            num += 1
            directory = os.path.join(root_dir, basedir, subdir)
            turk_dir = os.path.join(directory, domain + turk)
            turk_reg_dir = os.path.join(directory, domain + turk_reg)
            print(turk_dir, turk_reg_dir)
            with open(os.path.join(turk_dir, domain + '.json'), mode='rt') as file:
                classification = json.load(file, encoding='utf-8')
            with open(os.path.join(turk_reg_dir, domain + '.json'),
                      mode='rt')as file:
                regression = json.load(file, encoding='utf-8')
            print(classification[:20])
            print(regression[:20])

            # TODO add scores to res_all


if __name__ == '__main__':
    main()

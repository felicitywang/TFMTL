import json
import os
from pprint import pprint
DEBUG = True

TASKS = [
    'GOV_1000',
    'LIF_1000',
    'HEA_1000',
    'LAW_1000',
    'BUS_1000',
    'MIL_1000',
    'SPO_1000'
]

CLASS_SIZE = 2


def main():
    dataset_path_prefixes = [
        'min_1_max_-1_vocab_-1_doc_1000',
        'min_1_max_-1_vocab_-1_doc_-1',
        'min_1_max_-1_vocab_-1_doc_400_glove.6B.300d_expand',
        'min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand',
        'min_1_max_-1_vocab_-1_doc_-1_glove.6B.300d_expand',
        'min_1_max_-1_vocab_-1_doc_1000_wiki-news-300d-1M_expand',
        'min_1_max_-1_vocab_-1_doc_1000_GoogleNews-vectors-negative300-SLIM_expand'
    ]

    with open('encoders.json') as file:
        architectures = list(json.load(file).keys())
    pprint(architectures)

    for prefix in dataset_path_prefixes:
        for architecture in architectures:
            for cpu_or_gpu in ['cpu', 'gpu']:
                get_log(prefix, architecture, cpu_or_gpu)


def get_log(prefix, architecture, cpu_or_gpu):
    IN_DIR = 'results/seed_42/'
    OUT_DIR = 'logs'

    accs = {}
    f1s = {}

    for dataset in TASKS:
        name = dataset + '_' + prefix + '_' + architecture + '_' + cpu_or_gpu
        result_dir = os.path.join(IN_DIR, name)

        if not os.path.exists(result_dir):
            return

        if not os.path.exists(os.path.join(result_dir, 'log')):
            print(os.path.join(result_dir, 'e'))
            continue

        with open(os.path.join(result_dir, 'log')) as file:
            for line in file.readlines():
                if line.startswith('(*)'):
                    break
        for item in line.split():
            if 'Acc' in item:
                accs[dataset] = str(
                    float(item[item.find('=') + 1:item.rfind('**')]) * 100)
            elif 'F1' in item:
                f1s[dataset] = str(float(item[item.find('=') + 1:]) * 100)

    acc_list = []
    f1_list = []
    errors = []
    for dataset in TASKS:
        acc_list.append(accs.get(dataset, 'Error'))
        f1_list.append(f1s.get(dataset, 'Error'))
        if dataset not in accs:
            errors.append(dataset)

    with open(os.path.join(OUT_DIR, architecture + '_' + prefix), 'w') as file:
        file.write(' '.join(TASKS))
        file.write('\n')
        file.write(' '.join(acc_list))
        file.write('\n')
        file.write(' '.join(f1_list))
        file.write('\n')
        for error in errors:
            name = error + '_' + prefix + '_' + architecture + '_' + cpu_or_gpu
            result_dir = os.path.join(IN_DIR, name)
            # file.write(result_dir + '\n')
            file.write('ls ' + result_dir + '\n')
            file.write('cat ' + (os.path.join(result_dir, 'e') + '\n'))


if __name__ == "__main__":
    main()

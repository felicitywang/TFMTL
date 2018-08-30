"""Convert the gold labeled data to json

POS: 6 domains
NEG: other domains
"""

import gzip
import json
import os
from pprint import pprint

"""
base dir:
data/raw/gold/translations/{oracle, one, bop}/{1A, 1B}/{speech, text}

1A/domain_GOV.list (e.g. lines of file path like MATERIAL_BASE-1A_25144823)
the text would be in
1A/speech/MATERIAL_BASE-1A_25144823.txt or
1A/text/MATERIAL_BASE-1A_25144823.txt

'1A: GOV, LIF, BUS, LAW'
'1B: GOV, LIF, HEA, MIL'
"""

LABEL_DIR = 'data/raw/gold/labels'
RAW_BASE_DIR = 'data/raw/gold/translations'
RAW_SUB_DIRS = ['oracle', 'one', 'bop']

DOMAIN_6 = ['GOV', 'LIF', 'HEA', 'LAW', 'MIL', 'BUS']

LANG_DIRS = {
    '1A': ['GOV', 'LIF', 'BUS', 'LAW'],
    '1B': ['GOV', 'LIF', 'HEA', 'MIL']
}


def main():
    domain_dirs = {}

    for lang, domains in LANG_DIRS.items():
        for domain in domains:
            if domain not in domain_dirs:
                domain_dirs[domain] = [lang]
            else:
                domain_dirs[domain].append(lang)

    pos_filepaths = {
        'oracle': {},
        'one': {},
        'bop': {}
    }
    pos_nums = {
        'oracle': {},
        'one': {},
        'bop': {}
    }

    for subdir in RAW_SUB_DIRS:
        for lang in LANG_DIRS.keys():
            for domain in LANG_DIRS[lang]:
                pos_filename = os.path.join(LABEL_DIR, domain, lang,
                                            'domain_' + domain + '.list')
                tmp_pos_filepaths = []
                num = 0

                with open(pos_filename) as file:
                    print(pos_filename)
                    for line in file.readlines():
                        num += 1
                        for speech_or_text in ['speech', 'text']:
                            filepath = os.path.join(RAW_BASE_DIR, subdir,
                                                    lang,
                                                    speech_or_text,
                                                    line.strip() + '.txt')
                            if os.path.exists(filepath):
                                tmp_pos_filepaths.append(filepath)
                            else:
                                pass
                                # print('can\'t find ', filepath)
                if domain in pos_filepaths[subdir]:
                    pos_filepaths[subdir][domain].extend(tmp_pos_filepaths)
                    pos_nums[subdir][domain] += num
                else:
                    pos_filepaths[subdir][domain] = tmp_pos_filepaths
                    pos_nums[subdir][domain] = num

    # pprint(pos_filepaths)
    pprint(pos_nums)
    # for key, item in pos_filepaths.items():
    #   print(key, len(item))

    for subdir in RAW_SUB_DIRS:
        for domain in pos_nums[subdir]:
            assert pos_nums[subdir][domain] == len(pos_filepaths[subdir][domain]), \
                str(pos_nums[subdir][domain]) + ' ' + str(
                len(pos_filepaths[subdir][domain]))

    # make data.json

    all_filepaths = {
        'oracle': {},
        'one': {},
        'bop': {}
    }

    for subdir in RAW_SUB_DIRS:
        for domain, langs in domain_dirs.items():
            all_filepaths[subdir][domain] = get_all_filepaths(langs)
        for key, item in all_filepaths.items():
            print(key, len(item))

    pprint(all_filepaths)
    adfa

    # get gold data: pos + neg, index + text
    data = {}
    for domain in DOMAIN_6:
        data[domain] = []
        for filepath in all_filepaths[domain]:
            if filepath in pos_filepaths[domain]:
                label = 1
            else:
                label = 0
            data[domain].append({
                'text': read_file(filepath),
                'label': label
            })

    for key, item in data.items():
        print(key, len(item))

    # save data
    for domain in data:
        path = os.path.join('data/json/', domain + 'g', 'data.json.gz')
        with gzip.open(path, mode='wt') as file:
            json.dump(data[domain], file, ensure_ascii=False)

    # open for test
    for domain in data:
        path = os.path.join('data/json/', domain + 'g', 'data.json.gz')
        with gzip.open(path, mode='rt') as file:
            test = json.load(file)
            print(len(test))


def read_file(filepath):
    """Read lines of a file, convert to single line"""
    str = ""
    with open(filepath) as file:
        for line in file.readlines():
            str += line
    return str


def get_all_filepaths(dir, langs):
    all_filepaths = {}
    for subdir in RAW_SUB_DIRS:
        for lang in langs:
            for type in ['speech', 'text']:
                for filepath in os.listdir(os.path.join(RAW_BASE_DIR, subdir,
                                                        lang, type)):
                    if filepath.endswith('.txt'):
                        all_filepaths[subdir].append(
                            os.path.join(RAW_BASE_DIR, subdir, lang, type, filepath))
    return all_filepaths


def get_all_filepaths(dir, langs):
    all_filepaths = {}


if __name__ == '__main__':
    main()

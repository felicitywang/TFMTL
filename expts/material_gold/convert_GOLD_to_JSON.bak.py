"""Convert the gold labeled data to json

POS: 6 domains
NEG: other domains
"""

import gzip
import json
import os

"""
base dir:
/export/a05/mahsay/domain/goldstandard/

1A/domain_GOV.list (e.g. lines of file path like MATERIAL_BASE-1A_25144823)
the text would be in
1A/speech/MATERIAL_BASE-1A_25144823.txt or
1A/text/MATERIAL_BASE-1A_25144823.txt

'1A: GOV, LIF, BUS, LAW'
'1B: GOV, LIF, HEA, MIL'
"""

dir = "/export/a05/mahsay/domain/goldstandard/"


def main():
    DOMAIN_6 = ['GOV', 'LIF', 'HEA', 'LAW', 'MIL', 'BUS']

    language_dirs = {
        '1A': ['GOV', 'LIF', 'BUS', 'LAW'],
        '1B': ['GOV', 'LIF', 'HEA', 'MIL']
    }

    domain_dirs = {}
    for language in language_dirs:
        for domain in language_dirs[language]:
            if domain not in domain_dirs:
                domain_dirs[domain] = [language]
            else:
                domain_dirs[domain].append(language)

    pos_filepaths = {}
    pos_nums = {}

    for language in ['1A', '1B']:
        for domain in language_dirs[language]:
            filename = os.path.join(
                dir, language, 'domain_' + domain + '.list')
            filepaths = []
            # if not os.path.exists(filename):
            #   continue
            num = 0
            with open(filename) as file:
                print(filename)
                for line in file.readlines():
                    num += 1
                    for t in ['speech', 'text']:
                        filepath = os.path.join(dir,
                                                language,
                                                t,
                                                line.strip() + '.txt')
                        if os.path.exists(filepath):
                            filepaths.append(filepath)
                        else:
                            pass
                            # print('can\'t find ', filepath)
            if domain in pos_filepaths:
                pos_filepaths[domain].extend(filepaths)
                pos_nums[domain] += num
            else:
                pos_filepaths[domain] = filepaths
                pos_nums[domain] = num

    for key, item in pos_filepaths.items():
        print(key, len(item))

    for domain in pos_nums:
        assert pos_nums[domain] == len(pos_filepaths[domain]), \
            str(pos_nums[domain]) + ' ' + str(len(pos_filepaths[domain]))

    # make data.json

    all_filepaths = {}
    for domain, languages in domain_dirs.items():
        all_filepaths[domain] = get_all_filepaths(languages)
    for key, item in all_filepaths.items():
        print(key, len(item))

    # print(all_filepaths)

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


def get_all_filepaths(languages):
    all_filepaths = []
    for language in languages:
        for type in ['speech', 'text']:
            for filepath in os.listdir(os.path.join(dir, language, type)):
                if filepath.endswith('.txt'):
                    all_filepaths.append(os.path.join(
                        dir, language, type, filepath))
    return all_filepaths


if __name__ == '__main__':
    main()

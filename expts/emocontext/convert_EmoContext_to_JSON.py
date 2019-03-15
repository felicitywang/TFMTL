import pandas as pd
import json
import gzip


def main():
    train_list = pd.read_csv('data/raw/train.txt', sep='\t').to_dict('records')
    dev_list = pd.read_csv('data/raw/dev.txt', sep='\t').to_dict('records')

    label_dict = {
        'happy': 1,
        'angry': 3,
        'sad': 2,
        'others': 0
    }

    for i in train_list:
        i['text'] = i['turn1'] + ' ' + i['turn2'] + ' ' + i['turn1']
        i['label'] = label_dict[i['label']]
    for i in dev_list:
        i['text'] = i['turn1'] + ' ' + i['turn2'] + ' ' + i['turn1']
        i['label'] = label_dict[i['label']]

    all_list = []
    all_list.extend(train_list)
    all_list.extend(dev_list)

    train_index = list(range(len(train_list)))
    dev_index = list(range(len(train_list), len(train_list) + len(dev_list)))

    print(len(train_index))
    print(len(dev_index))

    index_dict = dict()
    index_dict['train'] = train_index
    index_dict['valid'] = dev_index
    index_dict['test'] = []

    assert len(set(index_dict['train']).intersection(index_dict['valid'])) == 0

    # TODO un-hard-code
    path = 'data/json/EmoContext/'

    with gzip.open(path + 'index.json.gz', mode='wt') as file:
        json.dump(index_dict, file, ensure_ascii=False)

    with gzip.open(path + 'data.json.gz', mode='wt') as file:
        json.dump(all_list, file, ensure_ascii=False)


if __name__ == '__main__':
    main()

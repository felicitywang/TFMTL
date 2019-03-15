import gzip
import json
import os
import sys


def main():
    dir = sys.argv[1]
    data = []
    for file in os.listdir(dir):
        if not file.endswith('txt'):
            continue
        with open(os.path.join(dir, file)) as fin:
            text = ''
            for line in fin.readlines():
                text += line
            data.append(
                {'text': text.strip(),
                 'id': file.replace('.txt', '')}
            )

    print(len(data))

    path = os.path.join(dir, 'data.json.gz')
    with gzip.open(path, mode='wt') as file:
        json.dump(data, file, ensure_ascii=False)

    with gzip.open(path, mode='rt') as file:
        data = json.load(file)
        print(len(data))


if __name__ == '__main__':
    main()

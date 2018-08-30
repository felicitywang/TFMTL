import gzip
import json
import os
import numpy as np
from statistics import mode

"""Get statistics"""


def print_statistics(data):
    # print('Min:', np.min(data))
    # print('Max:', np.max(data))
    # print('Avg:', np.average(data))
    # print('Median', np.median(data))
    # # print('Mode:', mode(data))
    # print()
    print('Min, Max, Avg, Median =', np.min(data),
          np.max(data), np.average(data), np.median(data))


def print_all():

    for filename in os.listdir('data/json/'):
        with gzip.open(os.path.join('data/json/', filename, 'data.bak.json.gz')) as file:
            print(filename)
            data = json.load(file)
            texts = [len(i['text']) for i in data]
            num_words = [len(i['text'].split()) for i in data]
            print('All:')
            print('Chars:', end='')
            print_statistics(texts)
            print('Words:', end='')
            print_statistics(num_words)

            print('Pos:')
            print('Chars:', end='')
            print_statistics(texts[:100])
            print('Words:', end='')
            print_statistics(num_words[:100])

            print('Neg:')
            print('Chars:', end='')
            print_statistics(texts[100:])
            print('Words:', end='')
            print_statistics(num_words[100:])

            print()


def print_single():

    for filename in os.listdir('data/json/'):
        with gzip.open(os.path.join('data/json/', filename, 'data.bak.json.gz')) as file:
            print(filename)
            data = json.load(file)
            texts = [len(i['text']) for i in data]
            num_words = [len(i['text'].split()) for i in data]
            print('All:')
            print('Chars:')
            print(texts)
            print('Words:')
            print(num_words)

            print('Pos:')
            print('Chars:')
            print(texts[:100])
            print('Words:')
            print(num_words[:100])

            print('Neg:')
            print('Chars:')
            print(texts[100:])
            print('Words:')
            print(num_words[100:])

            print()


if __name__ == '__main__':

    print_all()
    print_single()

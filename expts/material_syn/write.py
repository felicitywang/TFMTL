import os


def main():
    for dataset in os.listdir('data/json/'):
        print(
            'python ../scripts/write_tfrecords_single.py '
            + dataset
            + ' args_nopretrain.json')


if __name__ == '__main__':
    main()

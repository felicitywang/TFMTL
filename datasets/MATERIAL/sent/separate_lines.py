"""Separate the one line with multiple sentences to multiple lines"""

import os
import sys

from nltk import sent_tokenize


def main():
    dirs = sys.argv[1:]
    lengths = []
    for dir in dirs:
        for filename in os.listdir(dir):
            new_lines = []
            with open(os.path.join(dir, filename)) as fin:
                lines = fin.readlines()
                # assert len(lines) == 1, str(len(lines)) + ' ' + os.path.join(dir,
                #                                                              filename)
                for line in lines:
                    new_lines = sent_tokenize(line)
                    # print(len(new_lines), filename)
                    # print(new_lines)
                    # print('\n\n')
                    lengths.append(len(new_lines))
            with open(os.path.join(dir, filename), 'w') as fout:
                for line in new_lines:
                    fout.write(line + '\n')

    import pandas as pd
    lengths = pd.Series(lengths)
    print(lengths.value_counts())


if __name__ == '__main__':
    main()

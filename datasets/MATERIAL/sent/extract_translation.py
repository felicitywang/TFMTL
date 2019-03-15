import sys


def extract_translation(filepaths):
    for filepath in filepaths:
        new_lines = []
        with open(filepath) as file:
            for line in file.readlines():
                new_lines.append(line.strip().split('\t')[-1])
        with open(filepath + '.trans', 'w') as file:
            for line in new_lines:
                file.write(line + '\n')


def main():
    filepaths = []
    with open(sys.argv[1]) as file:
        for line in file.readlines():
            filepaths.append(line.strip())

    # for filepath in filepaths:
    #   if os.path.exists(filepath):
    #     print(filepath)

    extract_translation(filepaths)


if __name__ == '__main__':
    main()

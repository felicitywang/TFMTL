import random
import sys

vocab_size = 100
num_examples = 10
min_seq_length = 5
max_seq_length = 20
rules = dict()

rules[0] = lambda x: sum(x) % 2
rules[1] = lambda x: sum(x) % 4
rules[2] = lambda x: len(x)
# rules[1] = lambda x: 1 if sum(x) > (vocab_size / 2) else 0

def generate_data(dataset, outfile):
    rule = rules[dataset]

    with open(outfile, 'w') as f:
        for _ in range(num_examples):
            # generate sequence of integers (word IDs)
            seq = [random.randint(0, vocab_size) for _ in range(random.randint(min_seq_length, max_seq_length+1))]

            # determine its label
            label = rule(seq)

            # output to file
            f.write('{}\t{}\n'.format(seq, label))

def main():
    dataset_outfiles = sys.argv[1:]

    assert(len(dataset_outfiles) < len(rules))  # can only have as many datasets as there are rules to generate them

    for _ in range(num_examples):
        for dataset, outfile in enumerate(dataset_outfiles):
            generate_data(dataset, outfile)

if __name__ == '__main__':
    main()
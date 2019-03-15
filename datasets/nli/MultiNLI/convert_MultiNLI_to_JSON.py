import os
import json
import gzip
import sys

NLI_LABELS = ['contradiction', 'entailment', 'neutral']


def readMultinliData(datafolder="./data/", debug=True, num_instances=20):
    max_count = None
    if debug:
        max_count = num_instances + 1

    data_train = {"seq1": [],
                  "seq2": [],
                  "stance": [],
                  "genre": [],
                  "labels": []}
    data_train, _ = parseMultinliFile(os.path.join(datafolder,
                                                   'multinli_0.9/multinli_0.9/multinli_0.9_train.txt'),  # NOQA
                                      data_train,
                                      {},
                                      max_count,
                                      "train")
    data_dev = {"seq1": [], "seq2": [], "stance": [], "genre": [], "labels": []}
    data_test = {"seq1": [], "seq2": [], "stance": [], "genre": [], "labels": []}
    data_dev, data_test = parseMultinliFile(os.path.join(datafolder,
                                                         'multinli_0.9/multinli_0.9/multinli_0.9_dev_matched.txt'),
                                            # NOQA
                                            data_dev,
                                            data_test,
                                            max_count,
                                            "test")

    return data_train, data_dev, data_test


def parseMultinliFile(filepath, data_1, data_2, max_count, mode):
    reading_dataset = open(filepath, "r", encoding='utf-8')
    # The script reads into those lists.
    # If IDs for questions, supports or targets are defined, those are ignored.
    count = 0

    for line in reading_dataset:
        if max_count is None or count < max_count:
            lspl = line.strip("\n").split("\t")
            if len(lspl) == 15:
                gold_label, _, _, _, _, sentence1, sentence2, promptID, pairID, genre, _, _, _, _, _ = lspl  # NOQA
                if gold_label == "gold_label" or gold_label == "-":
                    continue
                data_dict = data_1
                if mode == "train" or (mode == "test" and count % 2 == 0):
                    data_dict = data_1
                elif mode == "test":
                    data_dict = data_2
                data_dict["seq1"].append(sentence1)
                data_dict["seq2"].append(sentence2)
                data_dict["stance"].append(gold_label)
                data_dict["genre"].append(genre)
                count += 1

    for lab in set(data_1["stance"]):
        data_1["labels"].append(lab)
    data_1["labels"] = sorted(data_1["labels"])
    assert data_1["labels"] == NLI_LABELS

    if data_2 != {}:
        for lab in set(data_2["stance"]):
            data_2["labels"].append(lab)
        data_2["labels"] = sorted(data_2["labels"])
        assert data_2["labels"] == NLI_LABELS

    return data_1, data_2


def make_example_list(d, starting_index):
    index = starting_index

    seq1_list = d["seq1"]
    seq2_list = d["seq2"]
    stance_list = d["stance"]

    examples = list(zip(*[seq1_list, seq2_list, stance_list]))

    res = []
    for example in examples:
        ex = dict()
        ex['index'] = index
        ex['seq1'] = example[0]
        ex['seq2'] = example[1]
        label = example[2]
        if label == "contradiction":
            label = 0
        elif label == "entailment":
            label = 1
        elif label == "neutral":
            label = 2
        else:
            raise ValueError("Unrecognized label: {}".format(label))
        ex['label'] = label
        res.append(ex)
        index += 1

    ending_index = index  # next example will have this index
    index_list = list(range(starting_index, ending_index))
    return res, ending_index, index_list


if __name__ == "__main__":
    datafolder = sys.argv[1]
    try:
        debug = True if (sys.argv[2].lower() == 'true') else False
        num_inst = int(sys.argv[3])
        print('Downsampling MultiNLI to {} examples.'.format(num_inst))
    except:
        debug = False
        num_inst = 20  # this value doesn't matter

    data_train, data_dev1, data_test = readMultinliData(datafolder=datafolder,
                                                        debug=debug,
                                                        num_instances=num_inst)

    index = 0
    train_list, index, train_index = make_example_list(data_train, index)
    dev_list, index, dev_index = make_example_list(data_dev1, index)
    test_list, index, test_index = make_example_list(data_test, index)

    index_dict = {
        'train': train_index,
        'valid': dev_index,
        'test': test_index
    }

    assert len(set(train_index).intersection(set(dev_index))) == 0
    assert len(set(train_index).intersection(set(test_index))) == 0
    assert len(set(dev_index).intersection(set(test_index))) == 0

    data_list = []
    data_list.extend(train_list)
    data_list.extend(dev_list)
    data_list.extend(test_list)

    # write out to JSON files
    #  index.json.gz
    with gzip.open(datafolder + 'index.json.gz', mode='wt') as file:
        json.dump(index_dict, file, ensure_ascii=False)
    #  data.json.gz
    with gzip.open(datafolder + 'data.json.gz', mode='wt') as file:
        json.dump(data_list, file, ensure_ascii=False)

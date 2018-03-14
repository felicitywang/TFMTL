import json
import sys

from mtl.util.dataset import Dataset
from mtl.util.util import make_dir

with open('args_' + sys.argv[1] + '.json', 'rt') as file:
    args_single = json.load(file)
    file.close()

json_dir = "data/json/" + sys.argv[1]

tfrecord_dir = "data/tf/single/"
tfrecord_dir += sys.argv[1] + "/"
tfrecord_dir += "min_" + str(args_single['min_frequency']) + \
                "_max_" + str(args_single['max_frequency']) + "/"
make_dir(tfrecord_dir)

dataset = Dataset(json_dir=json_dir,
                  tfrecord_dir=tfrecord_dir,
                  vocab_dir=tfrecord_dir,
                  max_document_length=args_single['max_document_length'],
                  min_frequency=args_single['min_frequency'],
                  max_frequency=args_single['max_frequency'],
                  train_ratio=args_single['train_ratio'],
                  valid_ratio=args_single['valid_ratio'],
                  subsample_ratio=args_single['subsample_ratio'],
                  padding=args_single['padding'],
                  write_bow=args_single['write_bow'],
                  write_tfidf=args_single['write_tfidf'],
                  generate_basic_vocab=False,
                  load_vocab=False,
                  generate_tf_record=True)

with open(tfrecord_dir + 'vocab_size.txt', 'w') as f:
    f.write(str(dataset.vocab_size))

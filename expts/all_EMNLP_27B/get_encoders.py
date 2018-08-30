"""Write encoders.json file"""
import codecs
import json
import os
from copy import deepcopy

from mtl.util.util import make_dir

debug = False  # use relative path to debug on laptop

if debug:
  root_dir = '../../'  # tfmtl
else:
  root_dir = '/export/a08/fwang/tfmtl/'

encoder_dict = {
  "serial_birnn_stock_glove_init_finetune": {
    "embedders_tied": True,
    "extractors_tied": True
  },
  "serial_birnn_stock_glove_init_freeze": {
    "embedders_tied": True,
    "extractors_tied": True
  },
  "serial_birnn_stock_glove_expand_finetune": {
    "embedders_tied": True,
    "extractors_tied": True
  },
  "serial_birnn_stock_glove_expand_freeze": {
    "embedders_tied": True,
    "extractors_tied": True
  },
  "serial_birnn_stock": {
    "embedders_tied": True,
    "extractors_tied": True
  },
}

no_pretrain_dict = {
  "embed_fn": "embed_sequence",
  "embed_kwargs": {
    "embed_dim": 100,
  },
  "extract_fn": "serial_lbirnn_stock",
  "extract_kwargs": {
    "num_layers": 1,
    "cell_type": "tf.contrib.rnn.LSTMCell",
    "cell_size": 100,
    "output_keep_prob": 1.0,
    "skip_connections": True,
    "initial_state_fwd": None,
    "initial_state_bwd": None
  }
}

init_dict = {
  "embed_fn": "init_glove",
  "embed_kwargs": {
    "embed_dim": 100,
    "glove_path": root_dir + "pretrained_word_embeddings/glove/glove.twitter.27B.100d.txt",
    "trainable": None,  # TODO
    "reverse_vocab_path": None,  # TODO
    # "data/tf/merged/Target_Topic2/glove.twitter.27B.100d_init/vocab_i2v.json",
    "random_size_path": None,  # TODO
    # "data/tf/merged/Target_Topic2/glove.twitter.27B.100d_init/random_size.txt",
  },
  "extract_fn": "serial_lbirnn_stock",
  "extract_kwargs": {
    "num_layers": 1,
    "cell_type": "tf.contrib.rnn.LSTMCell",
    "cell_size": 100,
    "output_keep_prob": 1.0,
    "skip_connections": True,
    "initial_state_fwd": None,
    "initial_state_bwd": None
  }
}

expand_dict = {
  "embed_fn": "expand_glove",
  "embed_kwargs": {
    "embed_dim": 100,
    "glove_path": root_dir + "pretrained_word_embeddings/glove/glove.twitter.27B.100d.txt",
    "trainable": None  # TODO
  },
  "extract_fn": "serial_lbirnn_stock",
  "extract_kwargs": {
    "num_layers": 1,
    "cell_type": "tf.contrib.rnn.LSTMCell",
    "cell_size": 100,
    "output_keep_prob": 1.0,
    "skip_connections": True,
    "initial_state_fwd": None,
    "initial_state_bwd": None
  }
}

RUDER_TASKS = [
  'Topic2',
  'Topic5',
  'Target',
  'Stance',
  'ABSA-L',
  'ABSA-R',
  'FNC-1',
  'MultiNLI'
]

if debug:
  data_dir = "data/tf/"
  encoder_dir = 'encoder_files'
else:
  data_dir = "/export/a08/fwang/tfmtl/expts/all_EMNLP_27B/data/tf/"
  encoder_dir = "/export/a08/fwang/tfmtl/expts/all_EMNLP_27B/encoder_files/"

make_dir(encoder_dir)

# single task

for task in ['st', 'mt']:
  for vocab in ['all_0', 'train_1']:
    for dataset in RUDER_TASKS:
      dataset_dir = os.path.join(data_dir, dataset + '-' + task)
      if task == 'st':
        reverse_vocab_path = os.path.join(dataset_dir, vocab +
                                          "_glove_init", dataset,
                                          "vocab_i2v.json")

        random_size_path = os.path.join(dataset_dir, vocab + "_glove_init",
                                        dataset,
                                        "random_size.txt")
      else:
        reverse_vocab_path = os.path.join(dataset_dir, vocab + "_glove_init",
                                          "vocab_i2v.json")

        random_size_path = os.path.join(dataset_dir, vocab + "_glove_init",
                                        "random_size.txt")
      # print(dataset_dir, reverse_vocab_path, random_size_path)

      # no pretrain
      encoder_dict["serial_birnn_stock"][
        dataset] = no_pretrain_dict

      # init fine tune
      encoder_dict["serial_birnn_stock_glove_init_finetune"][
        dataset] = deepcopy(init_dict)
      encoder_dict["serial_birnn_stock_glove_init_finetune"][dataset][
        "embed_kwargs"]["reverse_vocab_path"] = reverse_vocab_path
      encoder_dict["serial_birnn_stock_glove_init_finetune"][dataset][
        "embed_kwargs"]["random_size_path"] = random_size_path
      encoder_dict["serial_birnn_stock_glove_init_finetune"][dataset][
        "embed_kwargs"]["trainable"] = True

      # init freeze
      encoder_dict["serial_birnn_stock_glove_init_freeze"][
        dataset] = deepcopy(init_dict)
      encoder_dict["serial_birnn_stock_glove_init_freeze"][dataset][
        "embed_kwargs"]["reverse_vocab_path"] = reverse_vocab_path
      encoder_dict["serial_birnn_stock_glove_init_freeze"][dataset][
        "embed_kwargs"]["random_size_path"] = random_size_path
      encoder_dict["serial_birnn_stock_glove_init_freeze"][dataset][
        "embed_kwargs"]["random_size_path"] = random_size_path
      encoder_dict["serial_birnn_stock_glove_init_freeze"][dataset][
        "embed_kwargs"]["trainable"] = False

      # expand fine tune

      encoder_dict["serial_birnn_stock_glove_expand_finetune"][
        dataset] = deepcopy(expand_dict)
      encoder_dict["serial_birnn_stock_glove_expand_finetune"][
        dataset]["embed_kwargs"]["trainable"] = True

      # expand freeze
      encoder_dict["serial_birnn_stock_glove_expand_freeze"][
        dataset] = deepcopy(expand_dict)
      encoder_dict["serial_birnn_stock_glove_expand_freeze"][dataset][
        "embed_kwargs"]["trainable"] = False

    encoder_path = os.path.join(encoder_dir,
                                "encoder-" + task + "-" + vocab + ".json")
    with codecs.open(encoder_path, mode='w', encoding='utf-8') as file:
      json.dump(encoder_dict, file, ensure_ascii=False, indent=4)

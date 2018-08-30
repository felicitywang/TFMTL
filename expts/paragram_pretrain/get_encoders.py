"""Write encoders.json file"""
import codecs
import json
import os
from copy import deepcopy

from mtl.util.util import make_dir

debug = True  # use relative path to debug on laptop

if debug:
    root_dir = '../../'  # tfmtl
else:
    root_dir = '/export/a08/fwang/tfmtl/'

encoder_dict = {
    "serial_paragram_init_finetune": {
        "embedders_tied": True,
        "extractors_tied": True
    },
    "serial_paragram": {
        "embedders_tied": True,
        "extractors_tied": True
    },

}

init_dict = {
    "embed_fn": "init_glove",
    "embed_kwargs": {
        "embed_dim": 100,
        "glove_path": root_dir + "pretrained_word_embeddings/glove/glove.6B.100d.txt",
        "trainable": None,  # TODO
        "reverse_vocab_path": None,  # TODO
        # "data/tf/merged/Target_Topic2/glove.6B.100d_init/vocab_i2v.json",
        "random_size_path": None,  # TODO
        # "data/tf/merged/Target_Topic2/glove.6B.100d_init/random_size.txt",
    },
    "extract_fn": "serial_paragram",
    "extract_kwargs": {
        "reducer": "reduce_over_time",
        "apply_activation": True,
        "activation_fn": "tf.nn.tanh"
    }
}

no_pretrain_dict = {
    "embed_fn": "embed_sequence",
    "embed_kwargs": {
        "embed_dim": 100,
    },
    "extract_fn": "serial_paragram",
    "extract_kwargs": {
        "reducer": "reduce_over_time",
        "apply_activation": True,
        "activation_fn": "tf.nn.tanh"
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
    data_dir = "/export/a08/fwang/tfmtl/expts/all_EMNLP/data/tf/"
    encoder_dir = "/export/a08/fwang/tfmtl/expts/paragram_pretrain" \
                  "/encoder_files/"

make_dir(encoder_dir)

# single task

for task in ['mt']:
    for vocab in ['all_0']:
        for dataset in RUDER_TASKS:
            # no pretrain
            encoder_dict["serial_paragram"][
                dataset] = no_pretrain_dict

            dataset_dir = os.path.join(data_dir, dataset + '-' + task)
            reverse_vocab_path = os.path.join(dataset_dir, vocab + "_glove_init",
                                              "vocab_i2v.json")
            random_size_path = os.path.join(dataset_dir, vocab + "_glove_init",
                                            "random_size.txt")

            # init fine tune
            encoder_dict["serial_paragram_init_finetune"][
                dataset] = deepcopy(init_dict)
            encoder_dict["serial_paragram_init_finetune"][dataset][
                "embed_kwargs"]["reverse_vocab_path"] = reverse_vocab_path
            encoder_dict["serial_paragram_init_finetune"][dataset][
                "embed_kwargs"]["random_size_path"] = random_size_path
            encoder_dict["serial_paragram_init_finetune"][dataset][
                "embed_kwargs"]["trainable"] = True

        encoder_path = os.path.join(encoder_dir,
                                    "encoder-" + task + "-" + vocab + ".json")
        with codecs.open(encoder_path, mode='w', encoding='utf-8') as file:
            json.dump(encoder_dict, file, ensure_ascii=False, indent=4)

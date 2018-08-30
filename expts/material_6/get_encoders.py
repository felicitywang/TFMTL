'''Write encoders.json file for MATERIAL'''
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

MATERIAL_TASKS = [
    'GOV',
    'LIF',
    'BUS',
    'LAW',
    'HEA',
    'MIL'
]

encoder_dict = {
    'bilstm_expand_glove': {
        'embedders_tied': True,
        'extractors_tied': True
    },
    'bigru_expand_glove': {
        'embedders_tied': True,
        'extractors_tied': True
    },
    'paragram_expand_glove': {
        'embedders_tied': True,
        'extractors_tied': True
    },
    'cnn_expand_glove': {
        'embedders_tied': True,
        'extractors_tied': True
    },
    'bilstm_expand_fasttext': {
        'embedders_tied': True,
        'extractors_tied': True
    },
    'bigru_expand_fasttext': {
        'embedders_tied': True,
        'extractors_tied': True
    },
    'paragram_expand_fasttext': {
        'embedders_tied': True,
        'extractors_tied': True
    },
    'cnn_expand_fasttext': {
        'embedders_tied': True,
        'extractors_tied': True
    },
    'bilstm_expand_word2vec_slim': {
        'embedders_tied': True,
        'extractors_tied': True
    },
    'bigru_expand_word2vec_slim': {
        'embedders_tied': True,
        'extractors_tied': True
    },
    'paragram_expand_word2vec_slim': {
        'embedders_tied': True,
        'extractors_tied': True
    },
    'cnn_expand_word2vec_slim': {
        'embedders_tied': True,
        'extractors_tied': True
    },

}

bilstm_dict = {
    'embed_fn': 'expand_pretrained',
    'embed_kwargs': {
        'embed_dim': 100,
        'pretrained_path': None,  # TODO
        'trainable': True
    },
    'extract_fn': 'serial_lbirnn_stock',
    'extract_kwargs': {
        'num_layers': 1,
        'cell_type': 'tf.contrib.rnn.LSTMCell',
        'cell_size': 100,
        "cell_size": 100,
        "output_keep_prob": 0.5,
        # "output_keep_prob": 1.0,
        # "skip_connections": True,
        "attention": False,
        "attn_length": 3,
        "initial_state_fwd": None,
        "initial_state_bwd": None
    }
}

bigru_dict = {
    'embed_fn': 'expand_pretrained',
    'embed_kwargs': {
        'embed_dim': 100,
        'pretrained_path': None,  # TODO
        'trainable': True
    },
    'extract_fn': 'serial_lbirnn_stock',
    "extract_kwargs": {
        "num_layers": 1,
        "cell_type": "tf.contrib.rnn.GRUCell",
        "cell_size": 100,
        "output_keep_prob": 0.5,
        "attention": False,
        "attn_length": 3,
        "initial_state_fwd": None,
        "initial_state_bwd": None
    }
}

paragram_dict = {
    'embed_fn': 'expand_pretrained',
    'embed_kwargs': {
        'embed_dim': 100,
        'pretrained_path': None,  # TODO
        'trainable': True
    },
    'extract_fn': 'paragram',
    'extract_kwargs': {
        'reducer': 'reduce_max_over_time',
        'apply_activation': False,
        'activation_fn': None
    }

}

cnn_dict = {
    'embed_fn': 'expand_pretrained',
    'embed_kwargs': {
        'embed_dim': 100,
        'pretrained_path': None,  # TODO
        'trainable': True
    },
    'extract_fn': 'cnn_extractor',
    'extract_kwargs': {
        'num_filter': 128,
        'max_width': 7,
        'reducer': 'reduce_max_over_time',
        'activation_fn': 'tf.nn.relu'
    }
}

embedding_filepath_dict = {
    'glove': root_dir + 'pretrained_word_embeddings/' +
    'glove/glove.6B.100d.txt',
    'fasttext': root_dir + 'pretrained_word_embeddings/' +
    'fasttext/wiki-news-300d-1M.vec.zip',
    'word2vec_slim': root_dir + 'pretrained_word_embeddings/' +
    'word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz'
}

if debug:
    data_dir = 'data/tf/'
    encoder_dir = './'
else:
    data_dir = '/export/a08/fwang/tfmtl/expts/material_6/data/tf/'
    encoder_dir = '/export/a08/fwang/tfmtl/expts/material_6/'

make_dir(encoder_dir)

for dataset in MATERIAL_TASKS:
    dataset_dir = os.path.join(data_dir, dataset)

    encoder_dict['bilstm_expand_glove'][dataset] = deepcopy(bilstm_dict)
    encoder_dict['bilstm_expand_glove'][dataset]['embed_kwargs'][
        'pretrained_path'] = \
        embedding_filepath_dict['glove']

    encoder_dict['bigru_expand_glove'][dataset] = deepcopy(bilstm_dict)
    encoder_dict['bigru_expand_glove'][dataset]['embed_kwargs'][
        'pretrained_path'] = \
        embedding_filepath_dict['glove']

    encoder_dict['cnn_expand_glove'][dataset] = deepcopy(cnn_dict)
    encoder_dict['cnn_expand_glove'][dataset]['embed_kwargs'][
        'pretrained_path'] = \
        embedding_filepath_dict['glove']

    encoder_dict['paragram_expand_glove'][dataset] = deepcopy(paragram_dict)
    encoder_dict['paragram_expand_glove'][dataset]['embed_kwargs'][
        'pretrained_path'] = \
        embedding_filepath_dict['glove']

    encoder_dict['bilstm_expand_fasttext'][dataset] = deepcopy(bilstm_dict)
    encoder_dict['bilstm_expand_fasttext'][dataset]['embed_kwargs'][
        'pretrained_path'] = \
        embedding_filepath_dict['fasttext']

    encoder_dict['bigru_expand_fasttext'][dataset] = deepcopy(bilstm_dict)
    encoder_dict['bigru_expand_fasttext'][dataset]['embed_kwargs'][
        'pretrained_path'] = \
        embedding_filepath_dict['fasttext']

    encoder_dict['cnn_expand_fasttext'][dataset] = deepcopy(cnn_dict)
    encoder_dict['cnn_expand_fasttext'][dataset]['embed_kwargs'][
        'pretrained_path'] = \
        embedding_filepath_dict['fasttext']

    encoder_dict['paragram_expand_fasttext'][dataset] = deepcopy(paragram_dict)
    encoder_dict['paragram_expand_fasttext'][dataset]['embed_kwargs'][
        'pretrained_path'] = \
        embedding_filepath_dict['fasttext']

    encoder_dict['bilstm_expand_word2vec_slim'][dataset] = deepcopy(
        bilstm_dict)
    encoder_dict['bilstm_expand_word2vec_slim'][dataset]['embed_kwargs'][
        'pretrained_path'] = \
        embedding_filepath_dict['word2vec_slim']

    encoder_dict['bigru_expand_word2vec_slim'][dataset] = deepcopy(bilstm_dict)
    encoder_dict['bigru_expand_word2vec_slim'][dataset]['embed_kwargs'][
        'pretrained_path'] = \
        embedding_filepath_dict['word2vec_slim']

    encoder_dict['cnn_expand_word2vec_slim'][dataset] = deepcopy(cnn_dict)
    encoder_dict['cnn_expand_word2vec_slim'][dataset]['embed_kwargs'][
        'pretrained_path'] = \
        embedding_filepath_dict['word2vec_slim']

    encoder_dict['paragram_expand_word2vec_slim'][dataset] = deepcopy(
        paragram_dict)
    encoder_dict['paragram_expand_word2vec_slim'][dataset]['embed_kwargs'][
        'pretrained_path'] = \
        embedding_filepath_dict['word2vec_slim']

encoder_path = os.path.join(encoder_dir, 'encoders_100.json')
with codecs.open(encoder_path, mode='w', encoding='utf-8') as file:
    json.dump(encoder_dict, file, ensure_ascii=False, indent=4)

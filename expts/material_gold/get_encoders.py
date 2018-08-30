'''Write encoders.json file for MATERIAL'''
import codecs
import json
import os

DEBUG = False  # use relative path to debug on laptop
EMBED_DIM = 300
ATTN_LENGTH = 3
CELL_SIZE = 128
OUTPUT_KEEP_PROB = 0.5

if DEBUG:
    root_dir = '../../'  # tfmtl
else:
    root_dir = '/export/a08/fwang/tfmtl/'

MATERIAL_TASKS = [
    'GOV',
    'LIF',
    'BUS',
    'LAW',
    'HEA',
    'MIL',
    'SPO'
]

# extractors

extractors = {
    'meanpool_relu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_mean_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.relu'],
            'num_layers': 1
        }
    },
    'meanpool_linear': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_mean_over_time',
            'apply_activation': False,
            'activation_fns': [None],
            'num_layers': 1
        }
    },
    'meanpool_tanh': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_mean_over_time',
            'apply_activation': False,
            'activation_fns': ['tf.nn.tanh'],
            'num_layers': 1
        }
    },
    'meanpool_linear_relu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_mean_over_time',
            'apply_activation': False,
            'activation_fns': [None, 'tf.nn.relu'],
            'num_layers': 2
        }
    },
    'meanpool_relu_relu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_mean_over_time',
            'apply_activation': False,
            'activation_fns': ['tf.nn.relu', 'tf.nn.relu'],
            'num_layers': 1
        }
    },
    'maxpool_tanh': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_mean_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.tanh'],
            'num_layers': 1
        }
    },
    'maxpool_relu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_mean_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.relu'],
            'num_layers': 1
        }
    },
    'meanmax_linear': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': [None],
            'num_layers': 1
        }
    },
    'meanmax_relu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.relu'],
            'num_layers': 1
        }
    },
    'meanmax_elu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.elu'],
            'num_layers': 1
        }
    },
    'meanmax_selu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.selu'],
            'num_layers': 1
        }
    },
    'meanmax_tanh': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.tanh'],
            'num_layers': 1
        }
    },
    'meanmax_linear_linear': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': False,
            'activation_fns': [None, None],
            'num_layers': 2
        }
    },
    'meanmax_linear_relu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': False,
            'activation_fns': [None, 'tf.nn.relu'],
            'num_layers': 2
        }
    },
    'meanmax_linear_tanh': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': False,
            'activation_fns': [None, 'tf.nn.tanh'],
            'num_layers': 2
        }
    },
    'meanmax_linear_selu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': False,
            'activation_fns': [None, 'tf.nn.selu'],
            'num_layers': 2
        }
    },
    'meanmax_linear_eelu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': False,
            'activation_fns': [None, 'tf.nn.elu'],
            'num_layers': 2
        }
    },

    'meanmax_relu_relu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.relu', 'tf.nn.relu'],
            'num_layers': 2
        }
    },
    'meanmax_tanh_tanh': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.tanh', 'tf.nn.tanh'],
            'num_layers': 2
        }
    },
    'meanmax_tanh_relu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.tanh', 'tf.nn.relu'],
            'num_layers': 2
        }
    },
    'meanmax_relu_tanh': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.relu', 'tf.nn.tanh'],
            'num_layers': 2
        }
    },
    'bilstm': {
        'extract_fn': 'serial_lbirnn_stock',
        'extract_kwargs': {
            'num_layers': 1,
            'cell_type': 'tf.contrib.rnn.LSTMCell',
            'cell_size': CELL_SIZE,
            'output_keep_prob': OUTPUT_KEEP_PROB,
            'attention': False,
            'attn_length': None,
            'initial_state_fwd': None,
            'initial_state_bwd': None
        }
    },
    'bilstm_attention_3': {
        'extract_fn': 'serial_lbirnn_stock',
        'extract_kwargs': {
            'num_layers': 1,
            'cell_type': 'tf.contrib.rnn.LSTMCell',
            'cell_size': CELL_SIZE,
            'output_keep_prob': OUTPUT_KEEP_PROB,
            'attention': True,
            'attn_length': ATTN_LENGTH,
            'initial_state_fwd': None,
            'initial_state_bwd': None
        }
    },
    'bigru': {
        'extract_fn': 'serial_lbirnn_stock',
        'extract_kwargs': {
            'num_layers': 1,
            'cell_type': 'tf.contrib.rnn.GRUCell',
            'cell_size': CELL_SIZE,
            'output_keep_prob': OUTPUT_KEEP_PROB,
            'attention': False,
            'attn_length': None,
            'initial_state_fwd': None,
            'initial_state_bwd': None
        }
    },
    'bigru_attention_3': {
        'extract_fn': 'serial_lbirnn_stock',
        'extract_kwargs': {
            'num_layers': 1,
            'cell_type': 'tf.contrib.rnn.GRUCell',
            'cell_size': CELL_SIZE,
            'output_keep_prob': OUTPUT_KEEP_PROB,
            'attention': True,
            'attn_length': ATTN_LENGTH,
            'initial_state_fwd': None,
            'initial_state_bwd': None
        }
    },
    'cnn': {
        'extract_fn': 'cnn_extractor',
        'extract_kwargs': {
            'num_filter': 128,
            'max_width': 7,
            'reducer': 'reduce_max_over_time',
            'activation_fn': 'tf.nn.relu'
        }
    }

}

# embedders

embedding_filepath_dict = {
    'glove': os.path.join(
        root_dir, 'pretrained_word_embeddings/', 'glove/glove.6B.300d.txt'),
    'fasttext': os.path.join(
        root_dir, 'pretrained_word_embeddings/',
        'fasttext/wiki-news-300d-1M.vec.zip'),
    'word2vec_slim': os.path.join(
        root_dir, 'pretrained_word_embeddings/',
        'word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz')
}

embedders = {
    'nopretrain': {
        'embed_fn': 'embed_sequence',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM
        }
    },
    'fasttext_expand': {
        'embed_fn': 'expand_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['fasttext'],
            "trainable": True
        }
    },
    'glove_expand': {
        'embed_fn': 'expand_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['glove'],
            "trainable": True
        }
    },
    'word2vec_slim_expand': {
        'embed_fn': 'expand_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['word2vec_slim'],
            "trainable": True
        }
    }

}

encoders = {}

for embedder_name, embedder in embedders.items():
    for extractor_name, extractor in extractors.items():
        encoder_name = extractor_name + '_' + embedder_name
        encoder_dict = {
            'embedders_tied': False,
            'extractors_tied': True,
        }
        for task in MATERIAL_TASKS:
            encoder_dict[task] = {}
            encoder_dict[task] = {**embedder, **extractor}
        encoders[encoder_name] = encoder_dict

with codecs.open('encoders.json', mode='w', encoding='utf-8') as file:
    json.dump(encoders, file, ensure_ascii=False, indent=4)

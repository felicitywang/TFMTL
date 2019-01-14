# Copyright 2018 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific lang governing permissions and
# limitations under the License.
# =============================================================================

'''Generate encoder config file for qsub_stl_jobs.py

Usage:
    1. Modify embedders and extractors
    2. Run python get_qsub_encod.py [encod_cofig_name]('encoders.json' is
    not specified)
'''
import codecs
import json
import os
import sys

DEBUG = False  # use relative path to debug on laptop
EMBED_DIM = 100
ATTN_LENGTH = 3
CELL_SIZE = 100
OUTPUT_KEEP_PROB = 1.0

if DEBUG:
    root_dir = '../../'  # tfmtl
else:
    root_dir = '/export/a08/fwang/tfmtl/'

embedding_filepath_dict = {
    'glove': os.path.join(
        root_dir, 'pretrained_word_embeddings/', 'glove/glove.6B.100d.txt'),
    'fasttext': os.path.join(
        root_dir, 'pretrained_word_embeddings/',
        'fasttext/wiki-news-100d-1M.vec.zip'),
    'word2vec_slim': os.path.join(
        root_dir, 'pretrained_word_embeddings/',
        'word2vec/GoogleNews-vectors-negative100-SLIM.bin.gz'),
    'word2vec': os.path.join(
        root_dir, 'pretrained_word_embeddings/',
        'word2vec/GoogleNews-vectors-negative100.bin.gz'
    )
}

# extractors

extractors = {
    'dan_meanpool_relu_0.0': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'word_dropout_rate': 0.0,
            'reducer': 'reduce_mean_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.relu'],
            'num_layers': 1
        }
    },
    'dan_meanpool_relu_0.1': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'word_dropout_rate': 0.1,
            'reducer': 'reduce_mean_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.relu'],
            'num_layers': 1
        }
    },
    'dan_meanpool_linear': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_mean_over_time',
            'apply_activation': False,
            'activation_fns': [None],
            'num_layers': 1
        }
    },
    'dan_meanpool_tanh_0.0': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'word_dropout_rate': 0.0,
            'reducer': 'reduce_mean_over_time',
            'apply_activation': False,
            'activation_fns': ['tf.nn.tanh'],
            'num_layers': 1
        }
    },
    'dan_meanpool_tanh_0.1': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'word_dropout_rate': 0.1,
            'reducer': 'reduce_mean_over_time',
            'apply_activation': False,
            'activation_fns': ['tf.nn.tanh'],
            'num_layers': 1
        }
    },
    'dan_meanpool_linear_relu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_mean_over_time',
            'apply_activation': False,
            'activation_fns': [None, 'tf.nn.relu'],
            'num_layers': 2
        }
    },
    'dan_meanpool_relu_relu': {
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
    'dan_meanmax_linear': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': [None],
            'num_layers': 1
        }
    },
    'dan_meanmax_relu_0.0': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'word_dropout_rate': 0.0,
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.relu'],
            'num_layers': 1
        }
    },
    'dan_meanmax_relu_0.1': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'word_dropout_rate': 0.1,
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.relu'],
            'num_layers': 1
        }
    },
    'dan_meanmax_relu_0.3': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'word_dropout_rate': 0.3,
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.relu'],
            'num_layers': 1
        }
    },
    'dan_meanmax_relu_0.5': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'word_dropout_rate': 0.5,
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.relu'],
            'num_layers': 1
        }
    },
    'dan_meanmax_elu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.elu'],
            'num_layers': 1
        }
    },
    'dan_meanmax_selu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.selu'],
            'num_layers': 1
        }
    },
    'dan_meanmax_tanh_0.0': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'word_dropout_rate': 0.0,
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.tanh'],
            'num_layers': 1
        }
    },
    'dan_meanmax_tanh_0.1': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'word_dropout_rate': 0.0,
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.tanh'],
            'num_layers': 1
        }
    },
    'dan_meanmax_linear_linear': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': False,
            'activation_fns': [None, None],
            'num_layers': 2
        }
    },
    'dan_meanmax_linear_relu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': False,
            'activation_fns': [None, 'tf.nn.relu'],
            'num_layers': 2
        }
    },
    'dan_meanmax_linear_tanh': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': False,
            'activation_fns': [None, 'tf.nn.tanh'],
            'num_layers': 2
        }
    },
    'dan_meanmax_linear_selu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': False,
            'activation_fns': [None, 'tf.nn.selu'],
            'num_layers': 2
        }
    },
    'dan_meanmax_linear_eelu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': False,
            'activation_fns': [None, 'tf.nn.elu'],
            'num_layers': 2
        }
    },

    'dan_meanmax_relu_relu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.relu', 'tf.nn.relu'],
            'num_layers': 2
        }
    },
    'dan_meanmax_tanh_tanh': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.tanh', 'tf.nn.tanh'],
            'num_layers': 2
        }
    },
    'dan_meanmax_tanh_relu': {
        'extract_fn': 'dan',
        'extract_kwargs': {
            'reducer': 'reduce_over_time',
            'apply_activation': True,
            'activation_fns': ['tf.nn.tanh', 'tf.nn.relu'],
            'num_layers': 2
        }
    },
    'dan_meanmax_relu_tanh': {
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

embedders = {
    'nopretrain': {
        'embed_fn': 'embed_sequence',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM
        }
    },
    'fasttext_expand_finetune': {
        'embed_fn': 'expand_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['fasttext'],
            "trainable": True
        }
    },
    'fasttext_expand_nofinetune': {
        'embed_fn': 'expand_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['fasttext'],
            "trainable": False
        }
    },
    'glove_expand_finetune': {
        'embed_fn': 'expand_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['glove'],
            "trainable": True
        }
    },
    'glove_expand_finetune_proj_50': {
        'embed_fn': 'expand_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['glove'],
            "trainable": True,
            "proj_dim": 50
        }
    },
    'glove_expand_finetune_proj_100': {
        'embed_fn': 'expand_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['glove'],
            "trainable": True,
            "proj_dim": 100
        }
    },
    'glove_expand_nofinetune': {
        'embed_fn': 'expand_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['glove'],
            "trainable": False
        }
    },
    'glove_only_finetune': {
        'embed_fn': 'only_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['glove'],
            "trainable": True
        }
    },
    'glove_only_nofinetune': {
        'embed_fn': 'only_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['glove'],
            "trainable": False
        }
    },
    'glove_only_nofinetune_proj_100': {
        'embed_fn': 'only_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['glove'],
            "trainable": False,
            "proj": 100
        }
    },
    'glove_only_nofinetune_proj_150': {
        'embed_fn': 'only_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['glove'],
            "trainable": False,
            "proj": 150
        }
    },
    'glove_only_nofinetune_proj_200': {
        'embed_fn': 'only_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['glove'],
            "trainable": False,
            "proj": 200
        }
    },
    'word2vec_slim_expand_finetune': {
        'embed_fn': 'expand_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['word2vec_slim'],
            "trainable": True
        }
    },
    'word2vec_slim_expand_nofinetune': {
        'embed_fn': 'expand_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['word2vec_slim'],
            "trainable": False
        }
    },
    'word2vec_expand_finetune': {
        'embed_fn': 'expand_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['word2vec_slim'],
            "trainable": True
        }
    },
    'word2vec_expand_nofinetune': {
        'embed_fn': 'expand_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['word2vec_slim'],
            "trainable": False
        }
    },
    'fasttext_init_finetune': {
        'embed_fn': 'init_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['fasttext'],
            "trainable": True
        }
    },
    'fasttext_init_nofinetune': {
        'embed_fn': 'init_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['fasttext'],
            "trainable": False
        }
    },
    'glove_init_finetune': {
        'embed_fn': 'init_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['glove'],
            "trainable": True
        }
    },
    'glove_init_nofinetune': {
        'embed_fn': 'init_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['glove'],
            "trainable": False
        }
    },
    'word2vec_slim_init_finetune': {
        'embed_fn': 'init_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['word2vec_slim'],
            "trainable": True
        }
    },
    'word2vec_slim_init_nofinetune': {
        'embed_fn': 'init_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['word2vec_slim'],
            "trainable": False
        }
    },
    'word2vec_init_finetune': {
        'embed_fn': 'init_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['word2vec_slim'],
            "trainable": True
        }
    },
    'word2vec_init_nofinetune': {
        'embed_fn': 'init_pretrained',
        'embed_kwargs': {
            'embed_dim': EMBED_DIM,
            "pretrained_path": embedding_filepath_dict['word2vec_slim'],
            "trainable": False
        }
    }

}


def main():
    encoders = {}

    for embedder_name, embedder in embedders.items():
        for extractor_name, extractor in extractors.items():
            encoder_name = extractor_name + '_' + embedder_name
            encoders[encoder_name] = {**embedder, **extractor}
            encoders[encoder_name]['embed_kwargs_names'] = ' '.join(list(
                embedder['embed_kwargs'].keys()))
            encoders[encoder_name]['extract_kwargs_names'] = ' '.join(list(
                extractor['extract_kwargs'].keys()))

    print(encoders)

    filename = 'encoders.json' if len(sys.argv) == 1 else sys.argv[1]
    with codecs.open(filename, mode='w', encoding='utf-8') as file:
        json.dump(encoders, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()

#! /usr/bin/env python

# Copyright 2018 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# TODO build ARCHITECTURES dict from architectures.json and load from driver flags (?)
# TODO one encoder factory for all experiments (?)
# TODO move to mtl/encoders/ (?)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mtl.encoders.cnn import conv_and_pool
from mtl.encoders.paragram import paragram_phrase
from mtl.util.embed import embed_sequence
from mtl.util.reducers import *

ARCHITECTURES = {
    "cnn_LARGE_tied_word_embeddings":
        {"SSTb": {"num_filter": 64,
                  "max_width": 5,
                  "activation_fn": tf.nn.relu,
                  "reducer": reduce_max_over_time,
                  },
         "RTC": {"num_filter": 64,
                 "max_width": 5,
                 "activation_fn": tf.nn.relu,
                 "reducer": reduce_max_over_time,
                 },
         },

    "paragram_phrase_tied_word_embeddings":
        {
            "SSTb": {"reducer": reduce_max_over_time,
                     "apply_activation": False,
                     "activation_fn": None,
                     },
            "RTC": {"reducer": reduce_avg_over_time,
                    "apply_activation": False,
                    "activation_fn": None,
                    },
        },

    "rnn_rnn_untied":
        {"SSTb": {},
         "RTC": {},
         },

    "avg_cnn_and_cnn_fully_tied":
        {"num_filter": 128,
         "max_width": 5,
         "activation_fn": tf.nn.relu,
         "reducer": reduce_max_over_time,
         },
}


def encoder_graph(inputs, lengths, embed_fn, encode_fn):
    embed = embed_fn(inputs)
    return encode_fn(embed, lengths)


def build_prepared_encoders(vocab_size, args, encoder_hp=None):
    encoders = dict()

    if args.encoder_architecture == "cnn_LARGE_tied_word_embeddings":
        # Shared word embedding matrix for all datasets
        # Separate encoder for each dataset
        embed_temp = tf.make_template('embedding', embed_sequence,
                                      vocab_size=vocab_size,
                                      embed_dim=args.embed_dim)
        for ds in args.datasets:
            if (encoder_hp is not None) and (ds in encoder_hp):
                kwargs = encoder_hp[ds]
            else:
                kwargs = ARCHITECTURES[args.encoder_architecture][ds]
            encode_temp = tf.make_template('encoding_{}'.format(ds),
                                           conv_and_pool,
                                           kwargs)
            encoder = tf.make_template('encoder_{}'.format(ds),
                                       encoder_graph,
                                       embed_fn=embed_temp,
                                       encode_fn=encode_temp)
            encoders[ds] = encoder

        return encoders

    elif args.encoder_architecture == "paragram_phrase_tied_word_embeddings":
        # Shared word embedding matrix for all datasets
        # Separate encoder for each dataset
        embed_temp = tf.make_template('embedding', embed_sequence,
                                      vocab_size=vocab_size,
                                      embed_dim=args.embed_dim)
        for ds in args.datasets:
            if (encoder_hp is not None) and (ds in encoder_hp):
                kwargs = encoder_hp[ds]
            else:
                kwargs = ARCHITECTURES[args.encoder_architecture][ds]
            encode_temp = tf.make_template('encoding_{}'.format(ds),
                                           paragram_phrase,
                                           kwargs)
            encoder = tf.make_template('encoder_{}'.format(ds),
                                       encoder_graph,
                                       embed_fn=embed_temp,
                                       encode_fn=encode_temp)
            encoders[ds] = encoder

        return encoders

    elif args.encoder_architecture == "rnn_rnn_untied":
        # Separate word embedding matrix for each dataset
        # Separate encoder for each dataset
        raise NotImplementedError("rnn_rnn_untied encoder not implemented")

    elif args.encoder_architecture == "avg_cnn_and_cnn_fully_tied":
        # Shared word embedding matrix for all datasets
        # Shared encoder for all datasets
        if encoder_hp is not None:
            kwargs = encoder_hp
        else:
            kwargs = ARCHITECTURES[args.encoder_architecture]

        embed_temp = tf.make_template('embedding',
                                      embed_sequence,
                                      vocab_size=vocab_size,
                                      embed_dim=args.embed_dim)
        encode_temp = tf.make_template('encoding',
                                       conv_and_pool,
                                       kwargs)
        encoder = tf.make_template('encoder',
                                   encoder_graph,
                                   embed_fn=embed_temp,
                                   encode_fn=encode_temp)

        for ds in args.datasets:
            encoders[ds] = encoder

        return encoders

    else:
        raise ValueError("unrecognized encoder architecture: %s" %
                         (args.encoder_architecture))


def build_encoders(vocab_size, args, encoder_hp=None):
    encoders = dict()

    if args.encoder_architecture in ARCHITECTURES:
        encoders = build_prepared_encoders(
            vocab_size, args, encoder_hp=encoder_hp)

    else:
        raise NotImplementedError(
            "encoder architecture not supported: %s" % (args.encoder_architecture))

    return encoders

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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import json

from mtl.encoders.cnn import conv_and_pool
from mtl.encoders.paragram import paragram_phrase
from mtl.util.embed import embed_sequence
from mtl.util.reducers import *


def encoder_graph(inputs, lengths, embed_fn, encode_fn):
    return encode_fn(embed_fn(inputs), lengths)


def build_prepared_encoders(vocab_size, args, ARCHITECTURES, encoder_hp=None):
    encoders = dict()

    if args.encoder_architecture == "cnn_LARGE_tied_word_embeddings":
        # Shared word embedding matrix for all datasets
        # Separate encoder for each dataset
        embed_temp = tf.make_template('embedding', embed_sequence,
                                      vocab_size=vocab_size,
                                      embed_dim=args.word_embed_dim)
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
                                      embed_dim=args.word_embed_dim)
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
                                      embed_dim=args.word_embed_dim)
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

    elif args.encoder_architecture == "no_op":

        for ds in args.datasets:
            # embed_temp = tf.make_template('embedding',
            #                               no_op_embedding)
            # encode_temp = tf.make_template('encoding',
            #                                no_op_encoding)
            # encoder = tf.make_template('encoder',
            #                            encoder_graph,
            #                            embed_fn=embed_temp,
            #                            encode_fn=encode_temp)
            encoders[ds] = "no_op"

        return encoders

    else:
        raise ValueError("unrecognized encoder architecture: %s" %
                         args.encoder_architecture)


def replace_architecture_dict(args_dict):
    for k, i in args_dict.items():
        if type(i) is not str:
            continue
        if i == 'tf.nn.relu':
            args_dict[k] = tf.nn.relu
        elif i == 'reduce_avg_over_time':
            args_dict[k] = reduce_avg_over_time
        elif i == 'reduce_min_over_time':
            args_dict[k] = reduce_min_over_time
        elif i == 'reduce_max_over_time':
            args_dict[k] = reduce_max_over_time
        elif i == 'reduce_var_over_time':
            args_dict[k] = reduce_var_over_time
        else:
            raise ValueError("No such activation/reducer as %s" % i)


def parse_architectures(architectures_path):
    assert architectures_path is not None
    with codecs.open(architectures_path, mode='r', encoding='utf-8') as file:
        ARCHITECTURES = json.load(file)
        file.close()

    for encoder, encoder_args in ARCHITECTURES.items():
        if encoder_args is None:
            continue
        for k, v in encoder_args.items():
            if type(v) is dict:
                replace_architecture_dict(v)
            else:
                replace_architecture_dict(encoder_args)

    return ARCHITECTURES


def build_encoders(vocab_size, args, encoder_hp=None):
    ARCHITECTURES = parse_architectures(args.architectures_path)
    if args.encoder_architecture in ARCHITECTURES:
        encoders = build_prepared_encoders(
            vocab_size, args, ARCHITECTURES, encoder_hp=encoder_hp)
    else:
        raise NotImplementedError(
            "encoder architecture not supported: %s" % args.encoder_architecture)

    return encoders

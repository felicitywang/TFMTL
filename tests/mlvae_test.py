#! /usr/bin/env python

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
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from mtl.decoders import unigram
from mtl.encoders.cnn import conv_and_pool
from mtl.mlvae.mlvae_model import MultiLabel
from mtl.util.embed import embed_sequence


class MlvaeTests(tf.test.TestCase):
    def encoder_graph(self, inputs, embed_fn):
        embed = embed_fn(inputs)
        return conv_and_pool(embed)

    def decoder_graph(self, x, z, vocab_size):
        return unigram(x, z,
                       vocab_size=vocab_size)

    def build_encoders(self, vocab_size):
        encoders = dict()
        if self._share_embed:
            embed_temp = tf.make_template('embedding', embed_sequence,
                                          vocab_size=vocab_size,
                                          embed_dim=self._embed_dim)
            for ds in self._datasets:
                encoders[ds] = tf.make_template('encoder_{}'.format(ds), self.encoder_graph,
                                                embed_fn=embed_temp)
        else:
            for ds in self._datasets:
                embed_temp = tf.make_template('embedding_{}'.format(ds), tf.contrib.layers.embed_sequence,
                                              vocab_size=vocab_size,
                                              embed_dim=self._embed_dim)
                encoders[ds] = tf.make_template('encoder_{}'.format(ds), self.encoder_graph,
                                                embed_fn=embed_temp)

        return encoders

    def build_decoders(self, vocab_size):
        decoders = dict()
        if self._share_decoders:
            decoder = tf.make_template('decoder', self.decoder_graph,
                                       vocab_size=vocab_size)
            for ds in self._datasets:
                decoders[ds] = decoder
        else:
            for ds in self._datasets:
                decoders[ds] = tf.make_template('decoder_{}'.format(ds), self.decoder_graph,
                                                vocab_size=vocab_size)

        return decoders

    def setUp(self):
        self._N = 16
        self._batch_size = 4
        self._seq_len = 7
        self._vocab_size = 100
        self._share_embed = False
        self._embed_dim = 128
        self._datasets = ['foo']
        self._share_decoders = False
        self._inputs_key = "TEXT"
        self._labels_key = "LABELS"
        self._targets_key = "TARGETS"

    def generate_batch(self):
        batch = dict()
        batch[self._inputs_key] = tf.ones([self._batch_size, self._seq_len], dtype=tf.int32)
        batch[self._labels_key] = tf.ones([self._batch_size], dtype=tf.int32)
        batch[self._targets_key] = tf.ones([self._batch_size], dtype=tf.int32)
        return batch

    def test_model(self):
        class_sizes = dict()
        class_sizes['foo'] = 2

        dataset_order = ['foo']

        encoders = self.build_encoders(self._vocab_size)
        decoders = self.build_decoders(self._vocab_size)
        encoder_features = "bow"
        decoder_features = None

        m = MultiLabel(class_sizes=class_sizes,
                       dataset_order=dataset_order,
                       encoders=encoders,
                       decoders=decoders,
                       encoder_features=encoder_features,
                       decoder_features=decoder_features)

        batch = self.generate_batch()
        train_batches = {'foo': batch}

        with self.test_session() as sess:
            loss = m.get_multi_task_loss(train_batches)
            init_ops = [tf.global_variables_initializer(),
                        tf.local_variables_initializer()]
            sess.run(init_ops)
            loss_val = sess.run([loss])
            print('loss: {}'.format(loss_val))


if __name__ == '__main__':
    tf.test.main()

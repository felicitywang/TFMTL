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

import argparse

import tensorflow as tf

from mtl.util.encoder_factory import build_encoders

serial_lbirnn = True


class EncoderTests(tf.test.TestCase):
    def test_template(self):
        """Manually check that the variables created for various combinations
    of tied/untied embedders and extractors are correct."""

        parser = argparse.ArgumentParser()
        parser.add_argument('--architecture', default='example')
        parser.add_argument('--datasets', default=['SSTb', 'LMRD'])
        parser.add_argument('--encoder_config_file', default='tests/encoders.json')
        args = parser.parse_args()

        with self.test_session() as sess:
            if serial_lbirnn:
                encoders = build_encoders(args)
                inputs1 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
                lengths1 = tf.constant([3, 3, 3])

                inputs2 = tf.constant([[100, 101, 102, 103],
                                       [104, 105, 106, 107],
                                       [108, 109, 110, 111]])
                lengths2 = tf.constant([4, 4, 4])

                # indices = tf.constant([1,2,0])
                indices = None

                inputs = [inputs1, inputs2]
                lengths = [lengths1, lengths2]

                output_SSTb = encoders['SSTb'](inputs=inputs,
                                               lengths=lengths,
                                               is_training=True)
                output_LMRD = encoders['LMRD'](inputs=inputs,
                                               lengths=lengths,
                                               is_training=True)

                # output_SSTb = encoders['SSTb'](inputs=inputs,
                #                               lengths=lengths,
                #                               indices=indices)
                # output_LMRD = encoders['LMRD'](inputs=inputs,
                #                               lengths=lengths,
                #                               indices=indices)

                all_variables = tf.global_variables()
                trainable_variables = tf.trainable_variables()

                init_ops = [tf.global_variables_initializer(),
                            tf.local_variables_initializer()]
                sess.run(init_ops)

                all_var, train_var, s, l = sess.run([all_variables,
                                                     trainable_variables,
                                                     output_SSTb,
                                                     output_LMRD])

                print('Encoders: {}'.format(encoders))

                print('All variables created...')
                for var in all_variables:
                    print(var)

                print('Trainable variables created...')
                for var in trainable_variables:
                    print(var)

                print(output_SSTb.eval())

                print('SSTb output size: {}'.format(output_SSTb.get_shape().as_list()))
                print('LMRD output size: {}'.format(output_LMRD.get_shape().as_list()))

            else:
                encoders = build_encoders(args)

                inputs1 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
                lengths1 = tf.constant([3, 3, 3, 3])

                k = dict()

                # COMMENT THESE LINES OUT IF NOT USING L-BIRNN EXTRACTOR
                # indices1 = tf.constant([1, 1, 2, 0], dtype=tf.int64)
                # k = {'indices': indices1}
                #

                # COMMENT THESE LINES OUT IF NOT USING L-BIRNN or SERIAL L-BIRNN EXTRACTOR
                k['is_training'] = False
                #

                output_SSTb_1 = encoders['SSTb'](inputs=inputs1, lengths=lengths1, **k)
                output_LMRD_1 = encoders['LMRD'](inputs=inputs1, lengths=lengths1, **k)

                inputs2 = tf.constant([[1, 1, 1], [2, 2, 0]])
                lengths2 = tf.constant([3, 2])
                output_SSTb_2 = encoders['SSTb'](inputs=inputs2, lengths=lengths2)
                output_LMRD_2 = encoders['LMRD'](inputs=inputs2, lengths=lengths2)

                all_variables = tf.global_variables()
                trainable_variables = tf.trainable_variables()

                init_ops = [tf.global_variables_initializer(),
                            tf.local_variables_initializer()]
                sess.run(init_ops)

                all_var, train_var, s1, l1, s2, l2 = sess.run([all_variables,
                                                               trainable_variables,
                                                               output_SSTb_1,
                                                               output_LMRD_1,
                                                               output_SSTb_2,
                                                               output_LMRD_2])

                print('Encoders: {}'.format(encoders))

                print('All variables created...')
                for var in all_variables:
                    print(var)

                print('Trainable variables created...')
                for var in trainable_variables:
                    print(var)

                print(output_SSTb_1.eval())

                print('SSTb_1 size: {}'.format(output_SSTb_1.get_shape().as_list()))
                print('SSTb_2 size: {}'.format(output_SSTb_2.get_shape().as_list()))


if __name__ == '__main__':
    tf.test.main()

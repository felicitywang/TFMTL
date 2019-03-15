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

import tensorflow as tf


class EmbedTests(tf.test.TestCase):

    def test_template(self):
        def foo():
            with tf.variable_scope("foo_scope") as varscope:
                tmp = tf.get_variable("tmp", shape=[1])

                cell_fw = tf.contrib.rnn.LSTMCell(64,
                                                  initializer=tf.contrib.layers.xavier_initializer())
                cell_bw = tf.contrib.rnn.LSTMCell(64,
                                                  initializer=tf.contrib.layers.xavier_initializer())
                inputs = tf.constant([[[1.], [2.], [3.]], [[4.], [5.], [6.]]])
                lengths = tf.constant([3, 3])
                outputs, last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                      cell_bw,
                                                                      inputs,
                                                                      sequence_length=lengths,
                                                                      initial_state_fw=None,
                                                                      initial_state_bw=None,
                                                                      dtype=tf.float32)
                return tmp, outputs, last_state

        with self.test_session() as sess:
            with tf.variable_scope("seq1_scope") as varscope1:
                seq1_tmp, seq1_outputs, seq1_last_state = foo()

            with tf.variable_scope("seq2_scope") as varscope2:
                varscope1.reuse_variables()
                seq2_tmp, seq2_outputs, seq2_last_state = foo()

            all_variables = tf.global_variables()
            trainable_variables = tf.trainable_variables()

            init_ops = [tf.global_variables_initializer(),
                        tf.local_variables_initializer()]
            sess.run(init_ops)

            seq1_tmp_val, seq1_outputs_val, seq1_last_state_val, seq2_tmp_val, seq2_outputs_val, seq2_last_state_val = sess.run(
                [seq1_tmp, seq1_outputs, seq1_last_state, seq2_tmp,
                 seq2_outputs, seq2_last_state])

            all_var, train_var = sess.run([all_variables, trainable_variables])

            print('All variables created...')
            for var in all_variables:
                print(var)

            print('Trainable variables created...')
            for var in trainable_variables:
                print(var)


if __name__ == '__main__':
    tf.test.main()

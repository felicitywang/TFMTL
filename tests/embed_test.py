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

import numpy as np
import tensorflow as tf


class EmbedTests(tf.test.TestCase):
  def test_template(self):
    with self.test_session() as sess:
      temp1 = tf.make_template('embedding1', tf.contrib.layers.embed_sequence,
                               vocab_size=10,
                               embed_dim=2,
                               )
      temp2 = tf.make_template('embedding2', tf.contrib.layers.embed_sequence,
                               vocab_size=10,
                               embed_dim=2,
                               )

      embed1 = temp1(tf.constant([1, 4, 2, 6]))
      embed2 = temp1(tf.constant([1, 4, 2, 6]))
      embed3 = temp1(tf.constant([1, 1, 1, 1]))

      embed1_2 = temp2(tf.constant([1, 4, 2, 6]))

      init_ops = [tf.global_variables_initializer(),
                  tf.local_variables_initializer()]
      sess.run(init_ops)

      embed1_val, embed2_val, embed3_val, embed1_2_val = sess.run(
        [embed1, embed2, embed3, embed1_2])
      self.assertAlmostEqual(np.sum(embed1_val), np.sum(embed2_val))
      self.assertNotAlmostEqual(np.sum(embed1_val), np.sum(embed3_val))
      self.assertNotAlmostEqual(np.sum(embed1_val), np.sum(embed1_2_val))


if __name__ == '__main__':
  tf.test.main()

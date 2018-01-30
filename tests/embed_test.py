import tensorflow as tf
import numpy as np

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
            
            embed1 = temp1(tf.constant([1,4,2,6]))
            embed2 = temp1(tf.constant([1,4,2,6]))
            embed3 = temp1(tf.constant([1,1,1,1]))

            embed1_2 = temp2(tf.constant([1,4,2,6]))
            
            init_ops = [tf.global_variables_initializer(),
              tf.local_variables_initializer()]
            sess.run(init_ops)

            embed1_val, embed2_val, embed3_val, embed1_2_val = sess.run([embed1, embed2, embed3, embed1_2])
            self.assertAlmostEqual(np.sum(embed1_val), np.sum(embed2_val))
            self.assertNotAlmostEqual(np.sum(embed1_val), np.sum(embed3_val))
            self.assertNotAlmostEqual(np.sum(embed1_val), np.sum(embed1_2_val))
            
if __name__ == '__main__':
    tf.test.main()

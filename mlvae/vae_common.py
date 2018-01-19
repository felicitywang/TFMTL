from enum import Enum
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.init_ops import glorot_uniform_initializer
from tensorflow.python.ops.init_ops import zeros_initializer


class Inference(Enum):
  EXACT = 'sum'
  SAMPLE = 'sample'


def dense_layer(x, output_size, name, activation=None):
  return tf.layers.dense(x, output_size, name=name,
                         kernel_initializer=glorot_uniform_initializer(),
                         bias_initializer=zeros_initializer(),
                         activation=activation)


def cross_entropy_with_logits(logits, targets):
  log_q = tf.nn.log_softmax(logits)
  return -tf.reduce_sum(targets * log_q, 1)


def log_normal(x, mu, var, axis=-1):
  return -0.5 * tf.reduce_sum(tf.log(2.0 * np.pi) + tf.log(var) +
                              tf.square(x - mu) / var, axis)


def gaussian_sample(mean, var):
  sample = tf.random_normal(tf.shape(mean), mean, tf.sqrt(var))
  sample.set_shape(mean.get_shape())
  return sample


def get_tau(hp, decay=False):
  if decay:
    global_step = tf.train.get_global_step()
    return tf.maximum(hp.tau_min,
                      tf.train.natural_exp_decay(
                        hp.tau0,
                        global_step,
                        hp.tau_decay_steps,
                        hp.tau_decay_rate,
                        staircase=False))
  else:
    return tf.constant(hp.tau0)

#! /usr/bin/env python

# Copyright 2017 Johns Hopkins University. All Rights Reserved.
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
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from time import time

from dataset import Dataset
from input_dataset import InputDataset
from mlp import MLP
from optimizer import Optimizer
from util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", "yelp_data",
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_string('optimizer', 'adam', "Optimizer.")
flags.DEFINE_float('keep_prob', 0.75, "Dropout rate (keep probability)")
flags.DEFINE_float('lr', 0.00005, "Learning rate")
flags.DEFINE_float('l2_weight', 0.1, "L2 regularization weight")
flags.DEFINE_integer("num_components", 64,
                     "Number of mixture components for p(z)")
flags.DEFINE_integer("embed_dim", 128, "embedding dimension (VAE)")
flags.DEFINE_integer("latent_dim", 128, "latent dimension (VAE)")
flags.DEFINE_integer("seed", 42, "RNG seed")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("num_epoch", 10, "Epochs until LR is decayed.")
flags.DEFINE_integer('num_intra_threads', 0,
                     """Number of threads to use for intra-op
                     parallelism. If set to 0, the system will pick
                     an appropriate number.""")
flags.DEFINE_integer('num_inter_threads', 0,
                     """Number of threads to use for inter-op
                     parallelism. If set to 0, the system will pick
                     an appropriate number.""")
flags.DEFINE_boolean('force_gpu_compatible', True,
                     """whether to enable force_gpu_compatible in
                     GPU_Options""")

FLAGS = flags.FLAGS


def make_model(batch, num_classes, is_training):
    model = MLP(batch['bow'], batch['label'], num_classes=num_classes,
                dropout_rate=0.5, layers=[200, 200], is_training=is_training)
    return model


def run_epoch(sess, model, init_op=None, train_op=None):
    start_time = time()
    total_loss = 0
    total_correct = 0
    total_size = 0
    num_iter = 0
    fetches = {
        'loss': model.loss,
        'acc': model.accuracy,
        'correct': model.correct,
        'size': model.batch_size
    }
    if train_op is not None:
        fetches["train_op"] = train_op
    if init_op:
        sess.run(init_op)
    while True:
        try:
            vals = sess.run(fetches)
            # TODO
            # print(num_iter, vals['loss'], vals['acc'], vals['correct'],
            #       vals['size'])
            # print(num_iter, vals['loss'], vals['acc'])
            total_loss += vals['loss'] / vals['size']
            total_correct += vals['correct']
            total_size += vals['size']
            num_iter += 1
        except tf.errors.OutOfRangeError:
            break
    elapsed = time() - start_time
    total_acc = 1.0 * total_correct / total_size
    return total_loss, total_acc, elapsed


def log_result(split, result):
    total_loss, total_acc, elapsed = result
    hrs, mins, secs = hours_and_minutes_and_seconds(elapsed)
    logging.info("[%d hrs %d mins %d secs] %s: loss=%f acc=%f",
                 hrs, mins, secs, split, total_loss, total_acc)


def main(_):
    logging.set_verbosity(tf.logging.INFO)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    data_dir = "../datasets/sentiment/SSTb/"
    dataset = Dataset(data_dir=data_dir)
    num_classes = dataset.num_classes
    max_document_length = dataset.max_document_length
    vocab_size = dataset.vocab_size

    train_path, valid_path, test_path = dataset.train_path, \
                                        dataset.valid_path, \
                                        dataset.test_path

    features = {
        'word_id': tf.FixedLenFeature([max_document_length], dtype=tf.int64),
        'label': tf.FixedLenFeature([], dtype=tf.int64),
        'bow': tf.FixedLenFeature([vocab_size], dtype=tf.float32)
    }

    with tf.Graph().as_default():
        with tf.name_scope("Train"):
            dataset = InputDataset(train_path, features, FLAGS.batch_size)
            train_batch = dataset.batch
            train_init = dataset.init_op
            with tf.variable_scope("Model", reuse=None):
                m = make_model(train_batch, num_classes, is_training=True)
                opt = Optimizer()
                train_op = opt.optimize(m.loss)
        with tf.name_scope("Valid"):
            dataset = InputDataset(valid_path, features, FLAGS.batch_size)
            valid_batch = dataset.batch
            valid_init = dataset.init_op
            with tf.variable_scope("Model", reuse=True):
                mvalid = make_model(valid_batch, num_classes,
                                    is_training=False)
        with tf.name_scope("Test"):
            dataset = InputDataset(test_path, features, FLAGS.batch_size)
            test_batch = dataset.batch
            test_init = dataset.init_op
            with tf.variable_scope("Model", reuse=True):
                mtest = make_model(test_batch, num_classes, is_training=False)
        with tf.train.SingularMonitoredSession() as sess:
            for epoch in range(FLAGS.num_epoch):
                logging.info("Epoch %d", epoch)
                train_result = run_epoch(sess, m, init_op=train_init,
                                         train_op=train_op)
                log_result('train', train_result)
                valid_result = run_epoch(sess, mvalid, init_op=valid_init)
                log_result('valid', valid_result)
            test_result = run_epoch(sess, mtest, init_op=test_init)
            log_result('test', test_result)


def get_proto_config():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = FLAGS.num_intra_threads
    config.inter_op_parallelism_threads = FLAGS.num_inter_threads
    config.gpu_options.force_gpu_compatible = FLAGS.force_gpu_compatible
    return config


if __name__ == "__main__":
    tf.app.run()

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
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from time import time

from tasks.code.cnn import CNN
from tasks.code.dataset import Dataset
from tasks.code.input_dataset import InputDataset
from tasks.code.mlp import MLP
from tasks.code.optimizer import Optimizer
from tasks.code.util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_dir", "../datasets/sentiment/SSTb/",
                    "Where data.json.gz is loaded.")
flags.DEFINE_string("model_path", "./best_model/model.ckpt",
                    "Directory to save the best model checkpoint.")
flags.DEFINE_string('optimizer', 'adam', "Optimizer.")
flags.DEFINE_string('encoding', 'bow', "Encoding method of word ids.")
# flags.DEFINE_string('text_field_names', 'text', "Names of multiple text "
#                                                 "fields, separated by ' '")
# flags.DEFINE_string('label_field_name', 'label', "Names of the target column")
flags.DEFINE_float('dropout_rate', 0.5, "Dropout rate (1.0 - keep "
                                        "probability)")
flags.DEFINE_float('scale_ratio', 1.0, "Dropout rate (1.0 - keep "
                                       "probability)")
flags.DEFINE_float('lr', 0.0001, "Learning rate")
flags.DEFINE_float('l2_weight', 0.0001, "L2 regularization weight")
flags.DEFINE_integer("num_components", 64,
                     "Number of mixture components for p(z)")
flags.DEFINE_integer("embed_dim", 128, "embedding dimension (VAE)")
flags.DEFINE_integer("latent_dim", 128, "latent dimension (VAE)")
flags.DEFINE_integer("seed", 42, "RNG seed")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("num_epoch", 40, "Total number of training epochs.")
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
flags.DEFINE_integer("min_freq", 10, 'minimum frequency to build the '
                                     'vocabulary.')
flags.DEFINE_boolean("padding", True, 'whether to pad each sentence, should '
                                      'be True if encoding if bow')
flags.DEFINE_string("model", "mlp", "Which model to use: mlp/cnn")
FLAGS = flags.FLAGS


def make_model(batch, num_classes, vocab_size, is_training):
    if FLAGS.model == 'mlp':
        model = MLP(batch[FLAGS.encoding], batch['label'],
                    num_classes=num_classes,
                    dropout_rate=0.5,
                    layers=[100, 100],
                    is_training=is_training)
    elif FLAGS.model == 'cnn':
        model = CNN(batch[FLAGS.encoding], batch['label'],
                    num_classes=num_classes,
                    input_size=vocab_size,
                    # num_filter=128,
                    is_training=is_training)
    else:
        raise ValueError("No such model!")
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
    hrs, mins = hours_and_minutes(elapsed)
    logging.info("[%d hrs %d mins] %s: loss=%f acc=%f",
                 hrs, mins, split, total_loss, total_acc)


def main(_):
    logging.set_verbosity(tf.logging.INFO)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    dataset = Dataset(data_dir=FLAGS.data_dir, encoding=FLAGS.encoding,
                      # text_field_names=FLAGS.text_field_names,
                      # label_field_name=FLAGS.text_field_name,
                      min_frequency=FLAGS.min_freq,
                      valid_ratio=0.1, train_ratio=0.8, random_seed=42,
                      scale_ratio=FLAGS.scale_ratio, padding=True)

    num_classes = dataset.num_classes
    max_document_length = dataset.max_document_length
    vocab_size = dataset.vocab_size

    train_path, valid_path, test_path = dataset.train_path, \
                                        dataset.valid_path, \
                                        dataset.test_path

    features = {
        'label': tf.FixedLenFeature([], dtype=tf.int64)
    }

    if FLAGS.encoding == 'word_id':
        print(FLAGS.padding)
        if FLAGS.padding is True:
            features['word_id'] = tf.FixedLenFeature([max_document_length],
                                                     dtype=tf.int64)
        else:
            features['word_id'] = tf.VarLenFeature(dtype=tf.int64)
    elif FLAGS.encoding == 'bow':
        features['bow'] = tf.FixedLenFeature([vocab_size], dtype=tf.float32)
    else:
        raise ValueError("No such encoding!")

    with tf.Graph().as_default():
        with tf.name_scope("Train"):
            dataset = InputDataset(train_path, features, FLAGS.batch_size)
            train_batch = dataset.batch
            train_init = dataset.init_op
            with tf.variable_scope("Model", reuse=None):
                mtrain = make_model(train_batch, num_classes, vocab_size,
                                    is_training=True)
                opt = Optimizer()
                train_op = opt.optimize(mtrain.loss)
        with tf.name_scope("Valid"):
            dataset = InputDataset(valid_path, features, FLAGS.batch_size)
            valid_batch = dataset.batch
            valid_init = dataset.init_op
            with tf.variable_scope("Model", reuse=True):
                mvalid = make_model(valid_batch, num_classes, vocab_size,
                                    is_training=False)
        with tf.name_scope("Test"):
            dataset = InputDataset(test_path, features, FLAGS.batch_size)
            test_batch = dataset.batch
            test_init = dataset.init_op
            with tf.variable_scope("Model", reuse=True):
                mtest = make_model(test_batch, num_classes,
                                   vocab_size, is_training=False)

        saver = tf.train.Saver(var_list=tf.global_variables())
        with tf.train.SingularMonitoredSession() as session:
            # for epoch in range(FLAGS.num_epoch):
            #     logging.info("Epoch %d", epoch)
            #     train_result = run_epoch(sess, m, init_op=train_init,
            #                              train_op=train_op)
            #     log_result('train', train_result)
            #     valid_result = run_epoch(sess, mvalid, init_op=valid_init)
            #     log_result('valid', valid_result)

            best_epoch = 0
            max_valid_acc = 0.0
            last_valid_acc = 0.0
            decrease_time = 0

            for epoch in range(FLAGS.num_epoch):
                logging.info("Epoch %d", epoch)
                train_result = run_epoch(session, mtrain, init_op=train_init,
                                         train_op=train_op)
                log_result('train', train_result)
                valid_result = run_epoch(session, mvalid, init_op=valid_init)
                log_result('valid', valid_result)
                _, valid_acc, _ = valid_result
                # early stop if valid accuracy stops improving for 3
                # continuous times
                if valid_acc <= last_valid_acc:
                    decrease_time += 1
                    if decrease_time == 3:
                        break
                last_valid_acc = valid_acc
                if valid_acc > max_valid_acc:
                    best_epoch = epoch
                    max_valid_acc = valid_acc
                    # TODO model path
                    saver.save(session.raw_session(), FLAGS.model_path)

            logging.info(
                "Achieved max valid accuracy at epoch %d out of %d epochs",
                best_epoch + 1, FLAGS.num_epoch)

            with tf.Session() as session:
                saver.restore(session, FLAGS.model_path)
                test_result = run_epoch(session, mtest, init_op=test_init)
                log_result('test', test_result)


def get_proto_config():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = FLAGS.num_intra_threads
    config.inter_op_parallelism_threads = FLAGS.num_inter_threads
    config.gpu_options.force_gpu_compatible = FLAGS.force_gpu_compatible
    # maximun alloc gpu50% of MEM
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    return config


if __name__ == "__main__":
    tf.app.run()

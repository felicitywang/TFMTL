#! /usr/bin/env python3

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
import math
import random
import datetime
import json
from enum import Enum, unique
from six.moves import xrange

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.ops import parsing_ops

import tflm

from tflm.nets.unigram import unigram
from tflm.nets.ngram import ngram
from tflm.metrics.clustering import accuracy
from tflm.data import SymbolTable
from tflm.optim import EpochSummary
from tflm.corpora import TenSubreddits as Corpus
from tflm.data import InputDataset
from tflm.models import M12
from tflm.models.M12 import default_hparams as M12_hp
from tflm.models import M2
from tflm.models.M2 import default_hparams as M2_hp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.flags
logging = tf.logging

# Output paths
flags.DEFINE_string("data_path", "reddit_data", "Data directory")
flags.DEFINE_string('expt_path', 'reddit_expt', "Path to store run results.")
flags.DEFINE_string('run_id', None, "Run ID (subdir) to store results.")

# Data flags
flags.DEFINE_integer("max_vocab_size", 10000, "Vocabulary size.")
flags.DEFINE_integer("min_word_freq", -1, "Minimum word frequency.")
flags.DEFINE_integer("max_len", 200, "Maximum length for static batches.")

# Model hyper-parameters
flags.DEFINE_string("model", "M2", "Clustering model [M2 or M12]")
flags.DEFINE_string("decoder", "ngram", "Model to use for p(x | z)")
flags.DEFINE_integer("ngram_order", 3, "N-gram context window")
flags.DEFINE_integer("word_embed_dim", 256, "Size of word embedding.")
flags.DEFINE_string("inference", "sample", "One of `sample` or `sum`.")
flags.DEFINE_integer("K", 10,
                     "Number of mixture components for p(z)")
flags.DEFINE_boolean("condition_on_time", False,
                     "Condition on comment timestamps.")
flags.DEFINE_integer("embed_dim", 512,
                     "embedding dimension (encoder/decoder)")
flags.DEFINE_integer("latent_dim", 256, "latent dimension (VAE)")

# Optimization flags
flags.DEFINE_integer("num_epoch", 500, "Epochs until LR is decayed.")
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_boolean("check_numerics", False, "If we check for NaN.")
flags.DEFINE_string('optimizer', 'adam', "Optimizer.")
flags.DEFINE_float('lr', 0.0001, "Learning rate")
flags.DEFINE_float('max_grad_norm', 1.0, "Gradient norm threshold")

# Misc
flags.DEFINE_integer("seed", 42, "RNG seed")
flags.DEFINE_boolean("allow_soft_placement", True, "Allow soft placement.")
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
flags.DEFINE_boolean('log_device_placement', False,
                     "Log where ops are executed")

FLAGS = flags.FLAGS


@unique
class Example(Enum):
  BAG_OF_WORDS = 'bag_of_words'
  LENGTH = 'length'
  CONTEXT = 'context'
  TARGET = 'target'
  COUNT = 'count'
  LABEL = 'label'
  TIME = 'time'


def get_learning_rate(learning_rate):
  return tf.constant(learning_rate)


def get_train_op(tvars, grads, learning_rate, max_grad_norm, step):
  opt = tf.train.AdamOptimizer(learning_rate,
                               epsilon=1e-6,
                               beta1=0.85,
                               beta2=0.997)
  grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
  return opt.apply_gradients(zip(grads, tvars), global_step=step)


def bag_of_words(words, vocab_size, X, freq=False, norm=True,
                 dtype=np.float32):
  if type(words) != list:
    raise ValueError("words should be list")

  if len(words) < 1:
    raise ValueError("empty word sequence")

  if type(words[0]) != int:
    raise ValueError("must provide integer sequences")

  X.fill(0.0)
  if freq:
    for word in words:
      X[word] += 1
  else:
    types = set(words)
    for word in types:
      X[word] = 1

  if norm:
    denom = np.linalg.norm(X)
    denom += np.finfo(X.dtype).eps
    X = X / denom


def get_types_and_counts(token_list):
  counts = {x: token_list.count(x) for x in token_list}
  return counts.keys(), counts.values()


def write_features(examples, file_name, vocab_size, num_time_bins, sort=True):
  max_len = 0
  ntotal = 0
  total_len = 0.0

  # Storage for document vectors; this isn't sparse but should be
  X = np.zeros(vocab_size, np.float32)

  # Storage for timestamp features
  T = np.zeros(num_time_bins, np.float32)

  if os.path.exists(file_name):
    logging.info("feature file %s already exists; skipping preprocessing",
                 file_name)
    return

  # Look through all the examples
  with tf.python_io.TFRecordWriter(file_name) as writer:
    # Sorting by length speeds up training.
    if sort:
      logging.info("Sorting examples...".format(file_name))
      examples = sorted(examples, key=lambda x: len(x[Corpus.TEXT_KEY]))

    # Loop over all examples in the JSON
    logging.info("Writing examples to: %s", file_name)
    n = 0
    for example in examples:
      n += 1
      if n % 10000 == 0:
        logging.info("Processed %d examples", n)

      # Extract relevant fields
      body = example[Corpus.TEXT_KEY]
      subreddit = example[Corpus.LABEL_KEY]

      if FLAGS.decoder is "unigram":
        types, counts = get_types_and_counts(body)
        t = example[Corpus.TIME_KEY]
        assert len(types) == len(counts)
        assert len(types) > 0
        assert len(body) > 0, "empty example"

        # Sanity
        for t in types:
          assert t >= 0
          assert t < vocab_size
        for c in counts:
          assert c > 0
          assert c <= len(body)

        # Set the decoder
        targets = types
      elif FLAGS.decoder is "ngram":
        targets = body
      else:
        raise ValueError("unrecognized decoder: %s" % (FLAGS.decoder))

      # Keep track of the longest comment
      if len(targets) > max_len:
        max_len = len(targets)

      ntotal += 1
      total_len += float(len(targets))

      # Bag-of-words representation of document
      bag_of_words(body, vocab_size, X, freq=True, norm=True)
      bow = X.tolist()

      # Representation of the timestamp
      T.fill(0.0)
      for t in xrange(num_time_bins):
        T[t] = 1.0
      tvec = T.tolist()

      feature = {
        Example.TARGET.value: tf.train.Feature(
          int64_list=tf.train.Int64List(value=targets)),
        Example.LENGTH.value: tf.train.Feature(
          int64_list=tf.train.Int64List(value=[len(targets)])),
        Example.BAG_OF_WORDS.value: tf.train.Feature(
          float_list=tf.train.FloatList(value=bow)),
        Example.LABEL.value: tf.train.Feature(
          int64_list=tf.train.Int64List(value=[subreddit])),
        Example.TIME.value: tf.train.Feature(
          float_list=tf.train.FloatList(value=tvec))
      }

      if FLAGS.decoder is "unigram":
        feature[example.COUNT.value] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=counts))
      elif FLAGS.decoder is "ngram":
        context = []
        for target_t in range(len(targets)):
          ngram = [0] * FLAGS.ngram_order
          n = 1
          while target_t - n >= 0 and n <= FLAGS.ngram_order:
            ngram[-n] = targets[target_t - n]
            n += 1
          context += ngram
        feature[Example.CONTEXT.value] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=context))
      else:
        raise ValueError('unrecognized decoder: %s' % (FLAGS.decoder))

      record = tf.train.Example(features=tf.train.Features(
        feature=feature))
      writer.write(record.SerializeToString())
    # end of serialization loop

    mean_len = total_len / float(ntotal)
    logging.info("mean_len = {} max_len = {}".format(int(mean_len),
                                                     max_len))
    return max_len


def make_model(batch, decoders, is_training=True, log_file=None):
  if FLAGS.model.lower() == 'm12':
    hp = M12_hp()
  elif FLAGS.model.lower() == 'm2':
    hp = M2_hp()
  else:
    raise ValueError('unrecognized model: %s' % (FLAGS.model))

  fd = FLAGS.__flags
  hd = hp.values()

  for k, v in hd.items():
    if k in fd:
      if is_training:
        logging.info('  %s = %s', k, fd[k])
      hp.set_hparam(k, fd[k])

  if log_file:
    log_file.write("Command-line flags:\n")
    log_file.write(json.dumps(fd) + '\n')
    log_file.write("Model hyper-parameters:\n")
    log_file.write(json.dumps(hp.values()) + '\n')

  if FLAGS.decoder is "unigram":
    text_target = (batch[Example.TARGET.value],
                   batch[Example.COUNT.value],
                   batch[Example.LENGTH.value])
  elif FLAGS.decoder is "ngram":
    text_target = (batch[Example.TARGET.value],
                   batch[Example.CONTEXT.value],
                   batch[Example.LENGTH.value])
  else:
    raise ValueError("unrecognized decoder: %s" % (FLAGS.decoder))

  cond_vars = None
  if FLAGS.condition_on_time:
    logging.info("Conditioning on comment timestamps.")
    cond_vars = batch[Example.TIME.value]

  if FLAGS.model.lower() == 'm12':
    return M12(
      cond_vars=cond_vars,
      inputs=batch[Example.BAG_OF_WORDS.value],
      targets=text_target,
      decoders=decoders,
      is_training=is_training,
      hp=hp)
  elif FLAGS.model.lower() == 'm2':
    return M2(
      cond_vars=cond_vars,
      inputs=batch[Example.BAG_OF_WORDS.value],
      targets=text_target,
      decoders=decoders,
      is_training=is_training,
      hp=hp)
  else:
    raise ValueError("unrecognized model: %s" % (FLAGS.model))


def run_epoch(session, model, batch, num_words, writer=None,
              init_op=None, train_op=None, summary_op=None,
              check_op=None):

  summary = EpochSummary({
    'loss': 0.0,
    'num_words': 0.0,
    'num_iter': 0,
    'nll': 0.0,
    'labels': [],
    'gold_labels': []
  })
  summary.start_timer()

  fetches = {
    "loss": model.loss,
    "nll": model.nll,
    "num_words": num_words,
    "labels": model.labels,
    "gold_labels": batch[Example.LABEL.value]
  }
  if summary_op is not None:
    fetches["summary_op"] = summary_op
  if train_op is not None:
    fetches["train_op"] = train_op
  if check_op is not None:
    fetches["check_op"] = check_op

  if init_op:
    session.run(init_op)

  while True:
    try:
      vals = session.run(fetches)
      summary.increment_many(vals)
      summary.increment('num_iter', 1)
      if writer:
        global_step = tf.train.get_global_step()
        assert global_step, "couldn't find global step"
        global_step_value = tf.train.global_step(session, global_step)
        writer.add_summary(vals["summary_op"], global_step=global_step_value)
    except tf.errors.OutOfRangeError:
      break

  if summary['num_iter'] < 1:
    raise "Less than 1 iteration; something went wrong."

  summary.stop_timer()  # Stop the timer

  return summary


def log_result(split, epoch, lr, summary, writer=None,
               out_file=None, step=None):
  timestamp = datetime.datetime.now().strftime("%H:%M:%S")
  num_iter = summary['num_iter']
  wps = int(summary['num_words']/summary.elapsed)
  guess_labels = summary['labels']
  nclust = len(set(guess_labels))
  loss = summary['loss']/num_iter
  log_str = "[{}] {} {}: loss={:.2f} nclust={} wps={}".format(timestamp,
                                                              split,
                                                              epoch,
                                                              loss,
                                                              nclust,
                                                              wps)
  if step:
    log_str += " step={}".format(step)
  gold_labels = summary['gold_labels']
  assert len(gold_labels) == len(guess_labels)
  acc = accuracy(gold_labels, guess_labels)
  log_str += " accuracy=%.2f" % (acc)
  if writer:
    s = tf.Summary(value=[tf.Summary.Value(tag="nclust",
                                           simple_value=nclust)])
    writer.add_summary(s, global_step=step)
    s = tf.Summary(value=[tf.Summary.Value(tag="loss",
                                           simple_value=loss)])
    writer.add_summary(s, global_step=step)
    s = tf.Summary(value=[tf.Summary.Value(tag="accuracy",
                                           simple_value=acc)])
    writer.add_summary(s, global_step=step)

  logging.info(log_str)
  if out_file:
    out_file.write(log_str + '\n')


def preprocessing():
  # Where to serialize examples
  example_path = os.path.join(FLAGS.data_path, 'examples.tf')

  # Create corpus object (creates symbol tables).
  if FLAGS.min_word_freq < 1:
    min_word_freq = None
  else:
    min_word_freq = FLAGS.min_word_freq
  logging.info("Min word frequency: %s" % (str(min_word_freq)))

  if FLAGS.max_vocab_size < 0:
    max_vocab_size = None
  else:
    max_vocab_size = FLAGS.max_vocab_size
  logging.info("Max vocabulary size: %s" % (str(max_vocab_size)))

  corpus = Corpus(FLAGS.data_path,
                  min_word_freq=min_word_freq,
                  max_vocab_size=max_vocab_size)

  # Check if we've already done preprocessing.
  if os.path.exists(example_path):
    logging.info('feature extraction already done; skipping')
    return example_path, corpus.vocab, corpus.time_bin_map

  # ... otherwise write TF records now:
  max_len = write_features(corpus.examples(), example_path, len(corpus.vocab),
                           len(corpus.time_bin_map), sort=True)

  assert max_len <= FLAGS.max_len

  return example_path, corpus.vocab, corpus.time_bin_map


def main(_):
  logging.set_verbosity(tf.logging.INFO)
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  tf.set_random_seed(FLAGS.seed)

  # Prepare experiment directory
  if not os.path.exists(FLAGS.expt_path):
    os.mkdir(FLAGS.expt_path)

  # Prepare run directory
  if FLAGS.run_id is None:
    run_prefix = FLAGS.model + "_" + FLAGS.inference
    run_id = run_prefix + '_' + datetime.datetime.now().strftime('%m_%d_%H_%M')
  else:
    run_id = FLAGS.run_id
  run_path = os.path.join(FLAGS.expt_path, run_id)
  if os.path.exists(run_path):
    raise ValueError('Run path already exists: %s' % (run_path))
  logging.info("Run directory: %s", run_path)
  os.mkdir(run_path)

  # Create log file for evaluation results
  log_file_path = os.path.join(run_path, 'results.txt')
  logging.info("Log file path: %s", log_file_path)
  log_file = open(log_file_path, 'w')

  # Data preprocessing
  example_path, vocab, time_bin_map = preprocessing()
  vocab_size = len(vocab)
  num_time_bins = len(time_bin_map)
  logging.info("Vocabulary size: %d" % (vocab_size))
  logging.info("Number of time bins: %d" % (num_time_bins))

  # Features to read from serialized TF records
  FEATURES = {
    Example.TARGET.value: tf.VarLenFeature(dtype=tf.int64),
    Example.LENGTH.value: tf.FixedLenFeature([], dtype=tf.int64),
    Example.LABEL.value: tf.FixedLenFeature([], dtype=tf.int64),
    Example.TIME.value: tf.FixedLenFeature([num_time_bins], dtype=tf.float32),
    Example.BAG_OF_WORDS.value: tf.FixedLenFeature([vocab_size],
                                                   dtype=tf.float32),
  }

  if FLAGS.decoder is "unigram":
    FEATURES[Example.COUNT.value] = tf.VarLenFeature(dtype=tf.int64)
  elif FLAGS.decoder is "ngram":
    FEATURES[Example.CONTEXT.value] = tf.VarLenFeature(dtype=tf.int64)
  else:
    raise ValueError("unrecognized decoder: %s" % (FLAGS.decoder))

  # Create computation graph
  with tf.Graph().as_default() as g:
    # Decoders
    decoders = []
    if FLAGS.decoder is "unigram":
      decoders += [tf.make_template('text_decoder', unigram,
                                    vocab_size=vocab_size)]
    elif FLAGS.decoder is "ngram":
      decoders += [tf.make_template('text_decoder', ngram,
                                    vocab_size=vocab_size,
                                    embed_dim=FLAGS.word_embed_dim,
                                    ngram_order=FLAGS.ngram_order)]
    else:
      raise ValueError("unrecognized decoder: %s" % (FLAGS.decoder))

    # Global step
    global_step_tensor = tf.train.get_or_create_global_step()

    # Create input pipeline, model, and optimizer
    with tf.name_scope("Train"):
      dataset = InputDataset(example_path, FEATURES, FLAGS.batch_size)
      train_batch = dataset.batch
      train_init = dataset.init_op
      with tf.variable_scope("Model", reuse=None):
        m = make_model(train_batch, decoders, is_training=True,
                       log_file=log_file)

        if FLAGS.decoder is "unigram":
          num_words = tf.reduce_sum(train_batch[Example.COUNT.value])
        elif FLAGS.decoder is "ngram":
          num_words = tf.reduce_sum(train_batch[Example.LENGTH.value])
        else:
          raise ValueError("unrecognized decoder: %s" % (FLAGS.decoder))
        tvars, grads = m.get_var_grads()
        lr = get_learning_rate(FLAGS.lr)
        train_op = get_train_op(tvars, grads, lr, FLAGS.max_grad_norm,
                                global_step_tensor)

    # Print out total size of trainable parameters
    num_param = np.sum([np.prod(v.get_shape().as_list())
                        for v in tf.trainable_variables()])
    num_bytes = 4 * num_param
    num_mb = num_bytes / 1000000
    logging.info("Model size: %d MB", num_mb)

    # Combine all summaries
    tf.summary.scalar("lr", lr)
    summary_op = tf.summary.merge_all()

    # Log writer
    writer = tf.summary.FileWriter(run_path, graph=g)

    # Maybe check numerics.
    if FLAGS.check_numerics:
      check_op = tf.add_check_numerics_ops()
    else:
      check_op = None

    # Ops to initialize all variables.
    init_ops = [tf.global_variables_initializer(),
                tf.local_variables_initializer()]

    proto_config = get_proto_config(FLAGS)
    with tf.train.SingularMonitoredSession(config=proto_config) as sess:
      # Explicitly initialize all variables
      logging.info("Initializing all variables.")
      sess.run(init_ops)

      # Training
      logging.info("Starting training.")
      step = 0
      for epoch in xrange(FLAGS.num_epoch):
        train_result = run_epoch(sess, m, train_batch, num_words,
                                 init_op=train_init,
                                 train_op=train_op, writer=writer,
                                 summary_op=summary_op,
                                 check_op=check_op)
        step = tf.train.global_step(sess, global_step_tensor)
        log_result('train', epoch, sess.run(lr), train_result,
                   out_file=log_file, writer=writer, step=step)

      # Close log file
      log_file.close()


def get_proto_config(FLAGS):
  config = tf.ConfigProto()
  config.allow_soft_placement = FLAGS.allow_soft_placement
  config.intra_op_parallelism_threads = FLAGS.num_intra_threads
  config.inter_op_parallelism_threads = FLAGS.num_inter_threads
  config.gpu_options.force_gpu_compatible = FLAGS.force_gpu_compatible
  config.log_device_placement = FLAGS.log_device_placement
  return config


if __name__ == "__main__":
  tf.app.run()

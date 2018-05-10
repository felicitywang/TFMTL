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

# TODO tune alphas


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse as ap
import gzip
import json
import os
from time import time

import numpy as np
import tensorflow as tf
from six.moves import xrange
from tensorflow.contrib.training import HParams
from tqdm import tqdm

from mtl.models.mult import Mult
from mtl.util.clustering import aligned_accuracy
from mtl.util.metrics import accurate_number, metric2func
from mtl.util.pipeline import Pipeline
from mtl.util.util import make_dir

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging = tf.logging


def parse_args():
  p = ap.ArgumentParser()
  p.add_argument('--model', type=str,
                 help='Which model to use [mlvae|mult]')

  p.add_argument('--mode', choices=['train', 'test', 'predict', 'init'],
                 required=True,
                 help='Whether to train, test or predict. \n'
                      'train: train on train data and evaluate on valid data '
                      'for certain epochs, saving the best models\n'
                      'test: restore the saved model and evaluate on held-out '
                      'test data. \n'
                      'predict: restore the saved model and predict the '
                      'labels of the given text file. \n'
                      'init: restore the saved model and continue training.')
  p.add_argument('--experiment_name', default='', type=str,
                 help='Name of experiment.')
  p.add_argument('--tuning_metric', default='Acc', type=str,
                 help='Metric used to tune hyper-parameters')
  p.add_argument('--predict_tfrecord_path', type=str,
                 help='File path of the tf record file path of the text to '
                      'predict. Used in predict mode.')
  p.add_argument('--predict_dataset', type=str,
                 help='The dataset the text to predict belongs to.')
  p.add_argument('--predict_output_folder', type=str,
                 help='Folder to save the predictions using each model.')
  p.add_argument('--topics_paths', nargs='+', type=str,
                 help="""Paths to files mapping example index
                 to its topic (often data.json.gz)""")
  p.add_argument('--batch_size', default=128, type=int,
                 help='Size of batch.')
  p.add_argument('--eval_batch_size', default=128, type=int,
                 help='Size of evaluation batch.')
  p.add_argument('--word_embed_dim', default=128, type=int,
                 help='Word embedding size')
  p.add_argument('--share_decoders', action='store_true', default=False,
                 help='Whether decoders are shared across datasets')
  p.add_argument('--optimizer', default='adam', type=str,
                 help='Name of optimization algorithm to use')
  p.add_argument('--lr0', default=0.0001, type=float,
                 help='Initial learning rate')
  p.add_argument('--max_grad_norm', default=5.0, type=float,
                 help='Clip gradients to max_grad_norm during training.')
  p.add_argument('--num_train_epochs', default=50, type=int,
                 help='Number of training epochs.')
  p.add_argument('--print_trainable_variables', action='store_true',
                 default=False,
                 help='Diagnostic: print trainable variables')
  p.add_argument('--seed', default=42, type=int,
                 help='RNG seed')
  p.add_argument('--check_numerics', action='store_true', default=False,
                 help='Add operations to check for NaNs')
  p.add_argument('--num_intra_threads', default=20, type=int,
                 help="""Number of threads to use for intra-op
                     parallelism. If set to 0, the system will pick
                     an appropriate number.""")
  p.add_argument('--num_inter_threads', default=1, type=int,
                 help="""Number of threads to use for inter-op
                     parallelism. If set to 0, the system will pick
                     an appropriate number.""")
  p.add_argument('--log_device_placement', action='store_true',
                 default=False,
                 help='Log where compute graph is placed.')
  p.add_argument('--force_gpu_compatible', action='store_true',
                 default=False,
                 help='Throw error if any operations are not GPU-compatible')
  p.add_argument('--label_key', default="label", type=str,
                 help='Key for label field in the batches')
  p.add_argument('--input_key', default="tokens", type=str,
                 help='Key for input field in the batches')
  p.add_argument('--datasets', nargs='+', type=str,
                 help='Key of the dataset(s) to train and evaluate on')
  p.add_argument('--dataset_paths', nargs='+', type=str,
                 help="""Paths to the directory containing the TFRecord files
                  (train.tf, valid.tf, test.tf) for the dataset(s) given by 
                  the --datasets flag (in the same order)""")
  p.add_argument('--vocab_size_file', type=str,
                 help='Path to the file containing the vocabulary size')
  p.add_argument('--architecture', type=str,
                 help='Encoder architecture type (see encoder_factory.py for '
                      'supported architectures)')
  p.add_argument('--encoder_config_file', type=str,
                 help='Path of the args file of the architectures of the '
                      'experiment.')
  p.add_argument('--shared_hidden_dims', nargs='?', type=int,
                 default=[128, 128],
                 help='Sizes of the hidden layers shared by all datasets.')
  p.add_argument('--private_hidden_dims', nargs='?', type=int, default=None,
                 help='Sizes of the hidden layers private to each dataset.')
  p.add_argument('--shared_mlp_layers', type=int, default=2,
                 help='Number of hidden layers of the MLP model shared between'
                      ' datasets.')
  p.add_argument('--private_mlp_layers', type=int, default=0,
                 help='Number of hidden layers of the MLP model private for '
                      'each dataset.')
  p.add_argument('--num_filter', default=64, type=int,
                 help='Number of filters for the CNN model.')
  p.add_argument('--max_width', default=5, type=int,
                 help='Maximum window width for the CNN model.')
  p.add_argument('--alphas', nargs='+', type=float, default=[0.5, 0.5],
                 help='alpha for each dataset in the MULT model')
  p.add_argument('--class_sizes', nargs='+', type=int,
                 help='Number of classes for each dataset.')
  p.add_argument('--checkpoint_dir', type=str, default='./data/ckpt/',
                 help='Directory to save the checkpoints.')
  p.add_argument('--summaries_dir', type=str, default='./data/summ/',
                 help='Directory to save the Tensorboard summaries.')
  p.add_argument('--log_file', type=str,
                 help='File where results are stored.')
  p.add_argument('--input_keep_prob', type=float, default=1,
                 help="Probability to keep of the dropout layer before the MLP"
                      "(shared+private).")
  p.add_argument('--output_keep_prob', type=float, default=0.5,
                 help="Probability to keep of the dropout layer after the MLP"
                      "(shared+private).")
  p.add_argument('--token_lengths_key', type=str, default='tokens_length',
                 help='Key of the tokens lengths.')
  p.add_argument('--l2_weight', type=float, default=0.0,
                 help='Weight of the l2 regularization.')
  p.add_argument('--metrics', nargs='+', type=str, default=None,
                 help='Evaluation metrics for each dataset to use. '
                      'Supported metrics include:\n'
                      'Acc: accuracy score;\n'
                      'MAE_Macro: macro-averaged mean absolute error;\n'
                      'F1_Macro:  macro-averaged F1 score;\n'
                      'Recall_Macro: macro-averaged recall score.')

  return p.parse_args()


def get_num_records(tf_record_filename):
  c = 0
  for _ in tf.python_io.tf_record_iterator(tf_record_filename):
    c += 1
  return c


def get_vocab_size(vocab_file_path):
  with open(vocab_file_path, "r") as f:
    line = f.readline().strip()
    vocab_size = int(line)
  return vocab_size


def train_model(model,
                dataset_info,
                steps_per_epoch,
                args):
  """
  Train the model for certain epochs;
  Evaluate on valid data after each epoch;
  Save the model the performs the best on the validation epoch.
  """
  if args.mode not in ['train', 'init']:
    raise ValueError("train_model() called when in %s mode" % (args.mode))

  dataset_info, model_info = fill_info_dicts(dataset_info, args)

  train_batches = {name: model_info[name]['train_batch']
                   for name in model_info}
  additional_extractor_kwargs = dict()
  for dataset_name in model_info:
    additional_extractor_kwargs[dataset_name] = dict()
    with open(args.encoder_config_file, 'r') as f:
      extract_fn = json.load(f)[args.architecture][dataset_name]['extract_fn']
    if extract_fn == "serial_lbirnn":
      additional_extractor_kwargs[dataset_name]['is_training'] = True
      if args.experiment_name == "RUDER_NAACL_18":
        # use last token of last sequence as feature representation
        indices = train_batches[dataset_name][
          'seq2_length']  # TODO(seth): un-hard code this
        ones = tf.ones([tf.shape(indices)[0]], dtype=tf.int64)
        indices = tf.subtract(indices, ones)  # last token is at pos. length-1
        additional_extractor_kwargs[dataset_name]['indices'] = indices
    elif extract_fn == "lbirnn":
      additional_extractor_kwargs[dataset_name]['is_training'] = True
    elif extract_fn == "serial_lbirnn_stock":
      additional_extractor_kwargs[dataset_name]['is_training'] = True
    else:
      pass
  loss = model.get_multi_task_loss(train_batches,
                                   is_training=True,
                                   additional_extractor_kwargs=additional_extractor_kwargs)

  # # see if dropout and is_training working
  # # by checking train loss with different is_training the same

  # TODO loss on each dataset

  # Done building compute graph; set up training ops.

  # Training ops
  global_step_tensor = tf.train.get_or_create_global_step()
  zero_global_step_op = global_step_tensor.assign(0)
  lr = get_learning_rate(args.lr0)
  tvars, grads = get_var_grads(loss)
  train_op = get_train_op(tvars, grads, lr, args.max_grad_norm,
                          global_step_tensor, args.optimizer, name='train_op')
  init_ops = [tf.global_variables_initializer(),
              tf.local_variables_initializer()]
  config = get_proto_config(args)

  # Get training objective. The inputs are:
  #   1. A dict of { dataset_key: dataset_iterator }
  #

  fill_eval_loss_op(args, model, dataset_info, model_info)
  fill_pred_op_info(dataset_info, model, args, model_info)
  fill_topic_op(args, model_info)

  print("All the variables after defining valid/test accuracy:")
  all_variables = tf.global_variables()
  print(type(all_variables))
  for _ in all_variables:
    print(_)

  print("\n\n\n")

  # # Add ops to save and restore all the variables.

  # latest checkpoint
  # saves every several steps
  # automatically done by tf.train.SingularMonitorSession with
  # tf.train.CheckpoinSaverHook
  # TODO load from some checkpoint dif at the beginning(?)
  saver_hook = tf.train.CheckpointSaverHook(
    checkpoint_dir=os.path.join(args.checkpoint_dir, 'latest'),
    save_steps=100)

  # saved model builders for each model
  # builders = init_builders(args, model_info)

  saver = tf.train.Saver()

  with tf.train.SingularMonitoredSession(hooks=[saver_hook],
                                         config=config) as sess:
    # Initialize model parameters

    if args.mode == 'train':
      sess.run(init_ops)
    else:
      assert len(args.datasets) == 1
      checkpoint_path = model_info[args.datasets[0]]['checkpoint_path']
      print(checkpoint_path)
      saver.restore(sess, checkpoint_path)

    train_file_writer = tf.summary.FileWriter(
      os.path.join(args.summaries_dir, 'train'), graph=sess.graph)
    valid_file_writer = tf.summary.FileWriter(
      os.path.join(args.summaries_dir, 'valid'), graph=sess.graph)

    best_eval_performance = dict()
    for dataset_name in model_info:
      _train_init_op = model_info[dataset_name]['train_init_op']
      _valid_init_op = model_info[dataset_name]['valid_init_op']
      sess.run([_train_init_op, _valid_init_op])
      best_eval_performance[dataset_name] = {"epoch": -1,
                                             "acc": float('-inf'),
                                             "performance": None
                                             }

    best_total_acc = float('-inf')
    best_total_acc_epoch = -1

    # Do training
    with open(args.log_file, 'a') as f:
      f.write('VALIDATION RESULTS\n')
    for epoch in xrange(1, args.num_train_epochs + 1):
      start_time = time()

      total_acc = 0.0

      # Take steps_per_epoch gradient steps
      total_loss = 0
      num_iter = 0
      for _ in tqdm(xrange(steps_per_epoch)):
        step, loss_v, _ = sess.run(
          [global_step_tensor, loss, train_op])
        num_iter += 1
        total_loss += loss_v

        # loss_v is sum over a batch from each dataset of the average loss *per
        #  training example*
      assert num_iter > 0

      # average loss per batch (which is in turn averaged across examples)
      train_loss = float(total_loss) / float(num_iter)

      train_loss_summary = tf.Summary(
        value=[tf.Summary.Value(tag="loss", simple_value=train_loss)])
      train_file_writer.add_summary(train_loss_summary, global_step=step)

      # Evaluate held-out accuracy
      # if not args.test:  # Validation mode
      # Get performance metrics on each dataset
      for dataset_name in model_info:
        _pred_op = model_info[dataset_name]['valid_pred_op']
        _eval_labels = model_info[dataset_name]['valid_batch'][
          args.label_key]
        _eval_iter = model_info[dataset_name]['valid_iter']
        _get_topic_op = model_info[dataset_name]['valid_topic_op']
        _loss_op = model_info[dataset_name]['valid_loss_op']
        _metrics = compute_held_out_performance(sess,
                                                _pred_op,
                                                _eval_labels,
                                                _eval_iter,
                                                metrics=dataset_info[
                                                  dataset_name]['metrics'],
                                                labels=dataset_info[
                                                  dataset_name]['labels'],
                                                args=args,
                                                get_topic_op=_get_topic_op,
                                                topic_path=dataset_info[
                                                  dataset_name]['topic_path'],
                                                eval_loss_op=_loss_op)
        model_info[dataset_name]['valid_metrics'] = _metrics

      end_time = time()
      elapsed = end_time - start_time

      # Manually compute the validation loss since each dataset is iterated through once
      # in a serial manner and not "in parallel" (i.e., a batch from each)
      valid_loss = 0.0
      for (dataset_name, alpha) in zip(*[args.datasets, args.alphas]):
        valid_loss += float(alpha) * model_info[dataset_name]['valid_metrics'][
          'eval_loss']

      valid_loss_summary = tf.Summary(
        value=[tf.Summary.Value(tag="loss", simple_value=valid_loss)])
      valid_file_writer.add_summary(valid_loss_summary, global_step=step)
      main_task_acc = model_info[args.datasets[0]]['valid_metrics']['Acc']
      valid_main_task_accuracy_summary = tf.Summary(value=[
        tf.Summary.Value(tag="main-task-acc", simple_value=main_task_acc)])
      valid_file_writer.add_summary(valid_main_task_accuracy_summary,
                                    global_step=step)

      # Log performance(s)
      str_ = '[epoch=%d/%d step=%d (%d s)] train_loss=%s valid_loss=%s (per batch)' % (
        epoch, args.num_train_epochs, np.asscalar(step), elapsed,
        train_loss, valid_loss)

      for dataset_name in model_info:
        _num_eval_total = model_info[dataset_name]['valid_metrics'][
          'ntotal']
        # TODO use other metric here for tuning
        _eval_acc = model_info[dataset_name]['valid_metrics']['Acc']
        # _eval_align_acc = model_info[dataset_name]['valid_metrics'][
        #  'aligned_accuracy']

        str_ += '\n(%s) ' % (dataset_name)
        for m, s in model_info[dataset_name]['valid_metrics'].items():
          if m == args.tuning_metric:
            str_ += '*%s=%f* ' % (m, s)
          else:
            str_ += '%s=%f ' % (m, s)

        # Track best-performing epoch for each dataset
        if _eval_acc > best_eval_performance[dataset_name]["acc"]:
          best_eval_performance[dataset_name]["acc"] = _eval_acc
          best_eval_performance[dataset_name]["performance"] = \
            model_info[dataset_name]['valid_metrics'].copy()
          best_eval_performance[dataset_name]["epoch"] = epoch
          # save best model
          saver.save(sess.raw_session(),
                     model_info[dataset_name]['checkpoint_path'])

        total_acc += _eval_acc

      # Track best-performing epoch for collection of datasets
      if total_acc > best_total_acc:
        best_total_acc = total_acc
        best_total_acc_epoch = epoch
        best_epoch_results = str_
        if len(args.datasets) > 1:
          saver.save(sess.raw_session(),
                     os.path.join(args.checkpoint_dir, 'MULT',
                                  'model'))

      logging.info(str_)

      # Log dev results in a file
      with open(args.log_file, 'a') as f:
        f.write(str_ + '\n')

    print(best_eval_performance)
    print('Best total accuracy: {} at epoch {}'.format(best_total_acc,
                                                       best_total_acc_epoch))
    print(best_epoch_results)

    with open(args.log_file, 'a') as f:
      # f.write(best_eval_acc + '\n')
      # f.write('Best total accuracy: {} at epoch {}'.format(best_total_acc,
      #                                                     best_total_acc_epoch))
      f.write('\nBest single-epoch performance across all datasets\n')
      f.write(best_epoch_results + '\n\n')

    # Write (add) the result to a common report file
    with open(args.log_file, 'a') as f:
      for dataset in best_eval_performance.keys():
        f.write(str(dataset))
        f.write(" ")
      f.write("\n")
      for dataset, values in best_eval_performance.items():
        f.write(
          'Metrics on highest-accuracy epoch for dataset {}: {}\n'.format(
            dataset, values))

      f.write('Best total accuracy: {} at epoch {}\n\n'.format(best_total_acc,
                                                               best_total_acc_epoch))

    train_file_writer.close()
    valid_file_writer.close()


def test_model(model, dataset_info, args):
  """
  Evaluate on test data using the trained model.
  """
  dataset_info, model_info = fill_info_dicts(dataset_info, args)

  fill_eval_loss_op(args, model, dataset_info, model_info)
  fill_pred_op_info(dataset_info, model, args, model_info)
  fill_topic_op(args, model_info)

  str_ = '\nAccuracy on the held-out test data using different saved models:'

  model_names = args.datasets
  if len(args.datasets) > 1:
    model_names.append('MULT')

  saver = tf.train.Saver()

  for model_name in model_names:
    # load the saved best model
    str_ += '\nUsing the model that performs the best on (%s)' % model_name
    with tf.Session() as sess:
      if model_name == 'MULT':
        checkpoint_path = os.path.join(args.checkpoint_dir, 'MULT',
                                       'model')
      else:
        checkpoint_path = model_info[model_name]['checkpoint_path']
      print(checkpoint_path)

      saver.restore(sess, checkpoint_path)

      for dataset_name in model_info:
        _pred_op = model_info[dataset_name]['test_pred_op']
        _get_topic_op = model_info[dataset_name]['test_topic_op']
        _eval_labels = model_info[dataset_name]['test_batch'][
          args.label_key]
        _eval_iter = model_info[dataset_name]['test_iter']
        _metrics = compute_held_out_performance(sess,
                                                _pred_op,
                                                _eval_labels,
                                                _eval_iter,
                                                metrics=
                                                dataset_info[dataset_name][
                                                  'metrics'],
                                                labels=
                                                dataset_info[dataset_name][
                                                  'labels'],
                                                args=args,
                                                get_topic_op=_get_topic_op,
                                                topic_path=
                                                dataset_info[dataset_name][
                                                  'topic_path'],
                                                eval_loss_op=
                                                model_info[dataset_name][
                                                  'test_loss_op'])
        model_info[dataset_name]['test_metrics'] = _metrics

        _num_eval_total = model_info[dataset_name]['test_metrics'][
          'ntotal']
        _eval_acc = model_info[dataset_name]['test_metrics'][
          'Acc']
        # _eval_align_acc = model_info[dataset_name]['test_metrics'][
        #  'aligned_accuracy']
        str_ += '\n'
        if dataset_name == model_name:
          str_ += '(*)'
        else:
          str_ += '( )'
        str_ += '(%s)' % (dataset_name)
        for m, s in model_info[dataset_name]['test_metrics'].items():
          if m == args.tuning_metric:
            str_ += '*%s=%f* ' % (m, s)
          else:
            str_ += '%s=%f ' % (m, s)
        # str_ += '(%s) num_eval_total=%d eval_acc=%f eval_align_acc=%f' % (
        #  dataset_name,
        #  _num_eval_total,
        #  _eval_acc,
        #  _eval_align_acc)

  logging.info(str_)

  # Log test results in a file
  with open(args.log_file, 'a') as f:
    f.write('TEST RESULTS\n')
    f.write(str_ + '\n')


def predict(model, dataset_info, args):
  """
  Predict the text data using the trained model
  """
  dataset_info, model_info = fill_info_dicts(dataset_info, args)

  fill_pred_op_info(dataset_info, model, args, model_info)
  # fill_topic_op(args, model_info)

  str_ = 'Predictions of the given text data of dataset %s using different ' \
         'saved models:' % args.predict_dataset

  saver = tf.train.Saver()

  model_names = args.datasets
  if len(args.datasets) > 1:
    model_names.append('MULT')

  for model_name in model_names:
    # load the saved best model
    str_ += '\nUsing the model that performs the best on (%s)\n' % model_name

    with tf.Session() as sess:
      if model_name == 'MULT':
        checkpoint_path = os.path.join(args.checkpoint_dir, 'MULT',
                                       'model')
      else:
        checkpoint_path = model_info[model_name]['checkpoint_path']

      saver.restore(sess, checkpoint_path)

      dataset_name = args.predict_dataset
      _pred_op = model_info[dataset_name]['pred_pred_op']
      _pred_iter = model_info[dataset_name]['pred_iter']
      _predictions = get_all_predictions(sess, _pred_op, _pred_iter)

      str_ += str(_predictions)

      # TODO write to output file
      make_dir(args.predict_output_folder)
      with open(os.path.join(args.predict_output_folder, model_name) + '.pred',
                'w') as file:
        for i in _predictions:
          file.write(str(i) + '\n')
        file.close

  logging.info(str_)


def get_all_predictions(session, pred_op, pred_iterator):
  session.run(pred_iterator.initializer)

  predictions = []
  while True:
    try:
      pred_class = session.run(pred_op)
      pred_class_list = pred_class.tolist()
      predictions += pred_class_list
    except tf.errors.OutOfRangeError:
      break

  return predictions


def get_topic(batch):
  return batch['index']


def compute_held_out_performance(session,
                                 pred_op,
                                 eval_label,
                                 eval_iterator,
                                 metrics,
                                 labels,
                                 args,
                                 get_topic_op,
                                 topic_path,
                                 eval_loss_op):
  # pred_op: predicted labels
  # eval_label: gold labels

  # Initializer eval iterator
  session.run(eval_iterator.initializer)

  # # Accumulate predictions
  # ys = []
  # y_hats = []
  # while True:
  #   try:
  #     y, y_hat = session.run([eval_label, pred_op])
  #     assert y.shape == y_hat.shape, print(y.shape, y_hat.shape)
  #     y_list = y.tolist()
  #     y_hat_list = y_hat.tolist()
  #     ys += y_list
  #     y_hats += y_hat_list
  #   except tf.errors.OutOfRangeError:
  #     break
  #
  # assert len(ys) == len(y_hats)
  #
  # ntotal = len(ys)
  # ncorrect = 0
  # for i in xrange(len(ys)):
  #   if ys[i] == y_hats[i]:
  #     ncorrect += 1
  # acc = float(ncorrect) / float(ntotal)

  if topic_path != '' and topic_path is not None:
    with gzip.open(topic_path, mode='rt') as f:
      d = json.load(f, encoding='utf-8')
  index2topic = dict()
  for item in d:
    index2topic[item['index']] = item['seq1']

  # Accumulate predictions
  y_trues = []
  y_preds = []
  y_indexes = []
  y_topics = []
  total_eval_loss = 0
  num_eval_iter = 0
  while True:
    try:
      if args.experiment_name == "RUDER_NAACL_18":
        y_true, y_pred, y_index, eval_loss_v = session.run(
          [eval_label, pred_op, get_topic_op, eval_loss_op])
        num_eval_iter += 1
        total_eval_loss += eval_loss_v
        y_index = y_index.tolist()  # index of example in data.json
        y_topic = [index2topic[idx] for idx in
                   y_index]  # topic for each example so we can macro-average across topics
        y_indexes += y_index
        y_topics += y_topic
      else:
        y_true, y_pred, eval_loss_v = session.run(
          [eval_label, pred_op, eval_loss_op])
        num_eval_iter += 1
        total_eval_loss += eval_loss_v
      assert y_true.shape == y_pred.shape
      y_trues += y_true.tolist()
      y_preds += y_pred.tolist()
    except tf.errors.OutOfRangeError:
      break

  assert num_eval_iter > 0
  evaluation_loss = float(total_eval_loss) / float(num_eval_iter)

  # if args.experiment_name == "RUDER_NAACL_18":
  # for y_index, y_topic, y_t, y_p in zip(*[y_indexes, y_topics, y_trues, y_preds]):
  #  print('{} ({}): TRUE: {}, PRED: {}'.format(y_index, y_topic, y_t, y_p))

  ntotal = len(y_trues)
  ncorrect = accurate_number(y_trues=y_trues,
                             y_preds=y_preds,
                             labels=labels,
                             topics=y_topics)

  scores = dict()
  for metric in metrics:
    func = metric2func(metric)
    scores[metric] = func(y_trues, y_preds, labels, y_topics)

  res = dict()
  res['ntotal'] = ntotal
  res['ncorrect'] = ncorrect
  for score in scores:
    res[score] = scores[score]
  res['aligned_accuracy'] = aligned_accuracy(y_trues, y_preds)

  res['eval_loss'] = evaluation_loss
  return res
  # return {
  #  'ntotal': ntotal,
  #  'ncorrect': ncorrect,
  #  'accuracy': score,  # TODO score name
  #  'aligned_accuracy': aligned_accuracy(y_trues, y_preds),
  # }


def main():
  # Parse args
  args = parse_args()

  # Logging verbosity.
  logging.set_verbosity(tf.logging.INFO)

  # Seed numpy RNG
  np.random.seed(args.seed)

  # Path to each dataset
  dirs = dict()
  _dirs = zip(args.datasets, args.dataset_paths)
  for ds, path in _dirs:
    dirs[ds] = path

  topic_paths = dict()
  for ds, topic_path in zip(args.datasets, args.topics_paths):
    topic_paths[ds] = topic_path

  # Number of label types in each dataset
  class_sizes = dict()
  for dataset, class_size in zip(args.datasets, args.class_sizes):
    class_sizes[dataset] = class_size

  # Ordering of concatenation for input to decoder
  ordering = dict()
  for order, dataset in enumerate(args.datasets):
    ordering[dataset] = order

  # all the labels for each dataset
  # used only in some evaluation metrics(e.g. f1, recall)
  labels = dict()
  # if args.metrics in ['F1_Macro', 'Recall_Macro']:
  for dataset, dataset_path in zip(args.datasets, args.dataset_paths):
    with open(os.path.join(dataset_path, 'args.json')) as file:
      labels[dataset] = json.load(file)['labels']
      file.close()

  # evaluation metrics for each dataset
  metrics = dict()
  # if args.metrics == None:
  #  for dataset in args.datasets:
  #    metrics[dataset] = 'Acc'
  # else:
  #  assert len(args.metrics) == len(args.datasets)
  #  for dataset, metric in zip(args.datasets, args.metrics):
  #    metrics[dataset] = metric
  for dataset in args.datasets:
    metrics[dataset] = ['Acc', 'MAE_Macro', 'F1_Macro', 'Recall_Macro']

  # Read data
  dataset_info = dict()
  for dataset_name in args.datasets:
    dataset_info[dataset_name] = dict()
    # Collect dataset information/statistics
    dataset_info[dataset_name][
      'dataset_name'] = dataset_name  # feature name is just dataset name
    dataset_info[dataset_name]['dir'] = dirs[dataset_name]
    dataset_info[dataset_name]['topic_path'] = topic_paths[dataset_name]
    dataset_info[dataset_name]['class_size'] = class_sizes[dataset_name]
    dataset_info[dataset_name]['ordering'] = ordering[dataset_name]
    dataset_info[dataset_name]['labels'] = labels[dataset_name]
    dataset_info[dataset_name]['metrics'] = metrics[dataset_name]

    _dir = dataset_info[dataset_name]['dir']

    # Set paths to TFRecord files
    _dataset_train_path = os.path.join(_dir, "train.tf")
    dataset_info[dataset_name]['train_path'] = _dataset_train_path

    if args.mode in ['train', 'init']:
      _dataset_valid_path = os.path.join(_dir, "valid.tf")
      dataset_info[dataset_name]['valid_path'] = _dataset_valid_path
    elif args.mode == 'test':
      _dataset_test_path = os.path.join(_dir, "test.tf")
      dataset_info[dataset_name]['test_path'] = _dataset_test_path
    elif args.mode == 'predict':
      _dataset_predict_path = args.predict_tfrecord_path
      dataset_info[dataset_name]['pred_path'] = _dataset_predict_path
    else:
      raise ValueError('No such mode!')

  vocab_size = get_vocab_size(args.vocab_size_file)

  class_sizes = {dataset_name: dataset_info[dataset_name]['class_size'] for
                 dataset_name in dataset_info}

  order_dict = {dataset_name: dataset_info[dataset_name]['ordering'] for
                dataset_name in dataset_info}
  dataset_order = sorted(order_dict, key=order_dict.get)

  # This defines ALL the features that ANY model possibly needs access
  # to. That is, some models will only need a subset of these features.
  FEATURES = dict()
  for dataset, dataset_path in zip(args.datasets, args.dataset_paths):
    with open(os.path.join(dataset_path, 'args.json')) as f:
      json_config = json.load(f)
      text_field_names = json_config['text_field_names']
      for text_field_name in text_field_names:
        FEATURES[text_field_name + '_length'] = tf.FixedLenFeature([],
                                                                   dtype=tf.int64)
        if args.input_key == 'tokens':
          FEATURES[text_field_name] = tf.VarLenFeature(dtype=tf.int64)
        elif args.input_key == 'bow':
          FEATURES[text_field_name + '_bow'] = tf.FixedLenFeature([vocab_size],
                                                                  dtype=tf.float32)
        elif args.input_key == 'tfidf':
          FEATURES[text_field_name + '_tfidf'] = tf.FixedLenFeature(
            [vocab_size], dtype=tf.float32)
        else:
          raise ValueError("Input key %s not supported!" % (args.input_key))

  FEATURES['index'] = tf.FixedLenFeature([], dtype=tf.int64)
  if args.mode in ['train', 'test', 'init']:
    FEATURES['label'] = tf.FixedLenFeature([], dtype=tf.int64)

  # FEATURES = {
  #  'tokens_length': tf.FixedLenFeature([], dtype=tf.int64),
  #  # 'types': tf.VarLenFeature(dtype=tf.int64),
  #  # 'type_counts': tf.VarLenFeature(dtype=tf.int64),
  #  # 'types_length': tf.FixedLenFeature([], dtype=tf.int64),
  # }
  # if args.input_key == 'tokens':
  #  FEATURES['tokens'] = tf.VarLenFeature(dtype=tf.int64)
  # elif args.input_key == 'bow':
  #  FEATURES['bow'] = tf.FixedLenFeature([vocab_size], dtype=tf.float32)
  # elif args.input_key == 'tfidf':
  #  FEATURES['tfidf'] = tf.FixedLenFeature([vocab_size], dtype=tf.float32)
  # else:
  #  raise ValueError("Input key %s not supported!" % args.input_key)
  # if args.mode == 'train' or args.mode == 'test':
  #  FEATURES['label'] = tf.FixedLenFeature([], dtype=tf.int64)

  logging.info("Creating computation graph...")
  with tf.Graph().as_default() as graph:

    # Creating the batch input pipelines.  These will load & batch
    # examples from serialized TF record files.
    for dataset_name in dataset_info:
      _train_path = dataset_info[dataset_name]['train_path']
      ds = build_input_dataset(_train_path, FEATURES, args.batch_size,
                               is_training=True)
      dataset_info[dataset_name]['train_dataset'] = ds

      if args.mode in ['train', 'init']:
        # Validation dataset
        _valid_path = dataset_info[dataset_name]['valid_path']
        ds = build_input_dataset(_valid_path, FEATURES,
                                 args.eval_batch_size,
                                 is_training=False)
        dataset_info[dataset_name]['valid_dataset'] = ds
      elif args.mode == 'test':
        # Test dataset
        _test_path = dataset_info[dataset_name]['test_path']
        ds = build_input_dataset(_test_path, FEATURES,
                                 args.eval_batch_size,
                                 is_training=False)
        dataset_info[dataset_name]['test_dataset'] = ds
      elif args.mode == 'predict':
        _pred_path = dataset_info[dataset_name]['pred_path']
        ds = build_input_dataset(_pred_path, FEATURES,
                                 args.eval_batch_size,
                                 is_training=False)
        dataset_info[dataset_name]['pred_dataset'] = ds

    # This finds the size of the largest training dataset.
    training_files = [dataset_info[dataset_name]['train_path'] for
                      dataset_name
                      in dataset_info]
    max_N_train = max(
      [get_num_records(tf_rec_file) for tf_rec_file in training_files])

    # Seed TensorFlow RNG
    tf.set_random_seed(args.seed)

    # Maybe check for NaNs
    if args.check_numerics:
      logging.info("Checking numerics.")
      tf.add_check_numerics_ops()

    # Maybe print out trainable variables of the model
    if args.print_trainable_variables:
      for tvar in tf.trainable_variables():
        # TODO: also print and sum up all their sizes
        print(tvar)

    # Steps per epoch.
    # One epoch: all datasets have been seen completely at least once
    steps_per_epoch = int(max_N_train / args.batch_size)

    # Create model(s):
    # NOTE: models must support the following functions:
    #  * get_multi_task_loss()
    #        args: dictionary that maps dataset -> training batch
    #        returns: total loss accumulated over all batches in the dictionary
    #  * get_predictions()
    #        args: valid batch (with <batch_len> examples),
    #                 name of feature to predict
    #        returns: Tensor of size <batch_len> specifying the predicted label
    #                 for the specified feature for each of the examples
    #                 in the batch

    if args.model == 'mult':
      # hps = HParams()
      # hps.parse(path to hps.json)

      hps = set_hps(args)

      model = Mult(class_sizes=class_sizes,
                   dataset_order=dataset_order,
                   hps=hps,
                   args=args)

      # Do training
      if args.mode in ['train', 'init']:
        train_model(model,
                    dataset_info,
                    steps_per_epoch,
                    args)
      elif args.mode == 'test':
        test_model(model, dataset_info, args)
      elif args.mode == 'predict':
        # TODO text data to predict.tf
        predict(model, dataset_info, args)
      else:
        raise NotImplementedError(
          'Mode %s is not implemented!' % args.mode)

    else:
      raise ValueError("unrecognized model: %s" % args.model)


# TODO: use this directly:
# https://www.tensorflow.org/api_docs/python/tf/contrib/training/HParams
#   # If the hyperparameters are in json format use parse_json:
#   hparams.parse_json('{"learning_rate": 0.3, "activations": "relu"}')
def set_hps(args):
  hParams = HParams()

  for k, v in vars(args).items():
    hParams.add_hparam(k, v)

  return hParams
  #
  #
  # return HParams(hidden_dim=args.hidden_dim,
  #                num_filter=args.num_filter,
  #                max_width=args.max_width,
  #                word_embed_dim=args.word_embed_dim,
  #                alphas=args.alphas,
  #                label_key=args.label_key,
  #                input_key=args.input_key,
  #                dropout_rate=0.5,
  #                num_layers=args.num_layers,
  #                share_mlp=args.share_mlp
  #                )


def get_learning_rate(learning_rate):
  return tf.constant(learning_rate)


def get_train_op(tvars, grads, learning_rate, max_grad_norm, step, alg,
                 name=None):
  with tf.name_scope(name):
    if alg == "adam":
      opt = tf.train.AdamOptimizer(learning_rate,
                                   epsilon=1e-6,
                                   beta1=0.85,
                                   beta2=0.997)
    elif alg == "rmsprop":
      opt = tf.train.RMSPropOptimizer(learning_rate,
                                      decay=0.9,
                                      momentum=0.0,
                                      epsilon=1e-10,
                                      use_locking=False,
                                      centered=False)
    else:
      raise ValueError("unrecognized optimization algorithm: %s" % (alg))
    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    return opt.apply_gradients(zip(grads, tvars), global_step=step)


def get_proto_config(args):
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.log_device_placement = args.log_device_placement
  config.gpu_options.force_gpu_compatible = args.force_gpu_compatible
  config.intra_op_parallelism_threads = args.num_intra_threads
  config.inter_op_parallelism_threads = args.num_inter_threads
  return config


def fill_info_dicts(dataset_info, args):
  # Organizes inputs to the computation graph
  # The corresponding outputs are defined in train_model()

  # dataset_info: dict containing dataset-specific parameters/statistics
  #  e.g., number of classes, path to data
  # model_info: dict containing dataset-specific information w.r.t. running
  # the model
  #  e.g., data iterators, batches, prediction operations

  # use validation data for evaluation anyway
  if args.mode in ['train', 'init']:
    logging.info("Using validation data for evaluation.")
  elif args.mode == 'test':
    logging.info("Using test data for final evaluation.")
  elif args.mode == 'predict':
    logging.info("Predict on the given text data.")

  # Storage for pointers to dataset-specific Tensors
  model_info = dict()

  # paths to save/restore the checkpoints of the best model for each dataset
  for dataset_name in dataset_info:
    model_info[dataset_name] = dict()
    model_info[dataset_name]['checkpoint_path'] = os.path.join(
      args.checkpoint_dir, dataset_name, 'model')
  # Data iterators, etc.
  for dataset_name in dataset_info:

    if args.mode in ['train', 'init']:
      # Training data, iterator, and batch
      _train_dataset = dataset_info[dataset_name]['train_dataset']
      _train_iter = _train_dataset.iterator
      _train_init_op = _train_dataset.init_op
      _train_batch = _train_dataset.batch
      model_info[dataset_name]['train_iter'] = _train_iter
      model_info[dataset_name]['train_init_op'] = _train_init_op
      model_info[dataset_name]['train_batch'] = _train_batch

      # Held-out valid data, iterator, batch, and prediction operation
      _valid_dataset = dataset_info[dataset_name]['valid_dataset']
      _valid_iter = _valid_dataset.iterator
      _valid_init_op = _valid_dataset.init_op
      _valid_batch = _valid_dataset.batch
      model_info[dataset_name]['valid_iter'] = _valid_iter
      model_info[dataset_name]['valid_init_op'] = _valid_init_op
      model_info[dataset_name]['valid_batch'] = _valid_batch

    elif args.mode == 'test':
      _test_dataset = dataset_info[dataset_name]['test_dataset']
      _test_iter = _test_dataset.iterator
      _test_batch = _test_dataset.batch
      model_info[dataset_name]['test_iter'] = _test_iter
      model_info[dataset_name]['test_batch'] = _test_batch

    elif args.mode == 'predict':
      _pred_dataset = dataset_info[dataset_name]['pred_dataset']
      _pred_iter = _pred_dataset.iterator
      _pred_batch = _pred_dataset.batch
      model_info[dataset_name]['pred_iter'] = _pred_iter
      model_info[dataset_name]['pred_batch'] = _pred_batch

  def _create_feature_dict(ds, dataset_info, model_info):
    _feature_dict = dict()
    for dataset_name in dataset_info:
      if ds == dataset_name:
        # Observe the labels (from batch)
        _feature_dict[ds] = model_info[ds]['train_batch']
      else:
        # Don't observe the labels
        _feature_dict[ds] = None
    return _feature_dict

  # Create feature_dicts for each dataset
  if args.mode in ['train', 'init']:
    for dataset_name in model_info:
      model_info[dataset_name]['feature_dict'] = _create_feature_dict(
        dataset_name, dataset_info, model_info)

  # Return dataset_info dict and model_info dict
  return dataset_info, model_info


def fill_pred_op_info(dataset_info, model, args, model_info):
  # TODO(seth): refactor populating `additional_extractor_kwargs` into a function?
  additional_extractor_kwargs = dict()
  for dataset_name in model_info:
    additional_extractor_kwargs[dataset_name] = dict()
    with open(args.encoder_config_file, 'r') as f:
      extract_fn = json.load(f)[args.architecture][dataset_name]['extract_fn']
    if args.mode in ['train', 'init']:
      batch = model_info[dataset_name]['valid_batch']
    elif args.mode == 'test':
      batch = model_info[dataset_name]['test_batch']
    elif args.mode == 'predict':
      batch = model_info[dataset_name]['pred_batch']

    if extract_fn == "serial_lbirnn":
      additional_extractor_kwargs[dataset_name]['is_training'] = False
      if args.experiment_name == "RUDER_NAACL_18":
        # use last token of last sequence as feature representation
        indices = batch['seq2_length']  # TODO(seth): un-hard code this
        ones = tf.ones([tf.shape(indices)[0]], dtype=tf.int64)
        indices = tf.subtract(indices, ones)  # last token is at pos. length-1
        additional_extractor_kwargs[dataset_name]['indices'] = indices
    elif extract_fn == "lbirnn":
      additional_extractor_kwargs[dataset_name]['is_training'] = False
    elif extract_fn == "serial_lbirnn_stock":
      additional_extractor_kwargs[dataset_name]['is_training'] = False
    else:
      pass

  for dataset_name in model_info:
    if args.mode in ['train', 'init']:
      _valid_pred_op = model.get_predictions(
        model_info[dataset_name]['valid_batch'],
        dataset_name,
        dataset_info[dataset_name]['dataset_name'],
        additional_extractor_kwargs=additional_extractor_kwargs)
      model_info[dataset_name]['valid_pred_op'] = _valid_pred_op
    elif args.mode == 'test':
      _test_pred_op = model.get_predictions(
        model_info[dataset_name]['test_batch'],
        dataset_name,
        dataset_info[dataset_name]['dataset_name'],
        additional_extractor_kwargs=additional_extractor_kwargs)
      model_info[dataset_name]['test_pred_op'] = _test_pred_op
    elif args.mode == 'predict':
      _pred_pred_op = model.get_predictions(
        model_info[dataset_name]['pred_batch'],
        dataset_name,
        dataset_info[dataset_name]['dataset_name'],
        additional_extractor_kwargs=additional_extractor_kwargs)
      model_info[dataset_name]['pred_pred_op'] = _pred_pred_op


def fill_eval_loss_op(args, model, dataset_info, model_info):
  additional_extractor_kwargs = dict()
  for dataset_name in model_info:
    additional_extractor_kwargs[dataset_name] = dict()
    with open(args.encoder_config_file, 'r') as f:
      extract_fn = json.load(f)[args.architecture][dataset_name]['extract_fn']
    if args.mode in ['train', 'init']:
      batch = model_info[dataset_name]['valid_batch']
    elif args.mode == 'test':
      batch = model_info[dataset_name]['test_batch']

    if extract_fn == "serial_lbirnn":
      additional_extractor_kwargs[dataset_name]['is_training'] = False
      if args.experiment_name == "RUDER_NAACL_18":
        # use last token of last sequence as feature representation
        indices = batch['seq2_length']  # TODO(seth): un-hard code this
        ones = tf.ones([tf.shape(indices)[0]], dtype=tf.int64)
        indices = tf.subtract(indices, ones)  # last token is at pos. length-1
        additional_extractor_kwargs[dataset_name]['indices'] = indices
    elif extract_fn == "lbirnn":
      additional_extractor_kwargs[dataset_name]['is_training'] = False
    elif extract_fn == "serial_lbirnn_stock":
      additional_extractor_kwargs[dataset_name]['is_training'] = False
    else:
      pass

  for dataset_name in model_info:
    if args.mode in ['train', 'init']:
      _valid_loss_op = model.get_loss(
        model_info[dataset_name]['valid_batch'],
        dataset_name,
        dataset_info[dataset_name]['dataset_name'],
        additional_extractor_kwargs=additional_extractor_kwargs,
        is_training=False)
      model_info[dataset_name]['valid_loss_op'] = _valid_loss_op
    elif args.mode == 'test':
      _test_loss_op = model.get_loss(
        model_info[dataset_name]['test_batch'],
        dataset_name,
        dataset_info[dataset_name]['dataset_name'],
        additional_extractor_kwargs=additional_extractor_kwargs,
        is_training=False)
      model_info[dataset_name]['test_loss_op'] = _test_loss_op


def fill_topic_op(args, model_info):
  if args.experiment_name == "RUDER_NAACL_18":
    for dataset_name in model_info:
      if args.mode in ['train', 'init']:
        _valid_topic_op = get_topic(model_info[dataset_name]['valid_batch'])
        model_info[dataset_name]['valid_topic_op'] = _valid_topic_op
      elif args.mode == 'test':
        _test_topic_op = get_topic(model_info[dataset_name]['test_batch'])
        model_info[dataset_name]['test_topic_op'] = _test_topic_op
      elif args.mode == 'predict':
        _pred_topic_op = get_topic(model_info[dataset_name]['pred_batch'])
        model_info[dataset_name]['pred_topic_op'] = _pred_topic_op
  else:
    pass


def build_input_dataset(tfrecord_path, batch_features, batch_size,
                        is_training=True):
  if is_training:
    ds = Pipeline(tfrecord_path, batch_features, batch_size,
                  num_epochs=None,  # repeat indefinitely
                  )
  else:
    ds = Pipeline(tfrecord_path, batch_features, batch_size,
                  num_epochs=1)

  # We return the class because we might need to access the
  # initializer op for TESTING, while training only requires the
  # batches returned by the iterator.
  return ds


def get_var_grads(loss):
  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)
  return tvars, grads


if __name__ == "__main__":
  main()

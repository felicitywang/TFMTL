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

# TODO dataset name from sys arguments(?) -> one driver for all experiments
# TODO differentiate train/valid/test model (is_training) and add regularization
# TODO save best epoch
# TODO use encoder hyperparameters built from sys arguments
# TODO tune alphas


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse as ap
import os
from time import time

import numpy as np
import tensorflow as tf
from encoder_factory import build_encoders
from mtl.models.mult import Mult
from mtl.util.clustering import accuracy
from mtl.util.pipeline import Pipeline
from six.moves import xrange
from tensorflow.contrib.training import HParams
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging = tf.logging


def parse_args():
    p = ap.ArgumentParser()
    p.add_argument('--model', type=str,
                   help='Which model to use [mlvae|mult]')
    p.add_argument('--test', action='store_true', default=False,
                   help='Use held-out test data. WARNING: DO NOT TUNE ON TEST')
    p.add_argument('--batch_size', default=128, type=int,
                   help='Size of batch.')
    p.add_argument('--eval_batch_size', default=256, type=int,
                   help='Size of evaluation batch.')
    p.add_argument('--word_embed_dim', default=256, type=int,
                   help='Word embedding size')
    p.add_argument('--share_embed', action='store_true', default=False,
                   help='Whether datasets share word embeddings')
    p.add_argument('--share_decoders', action='store_true', default=False,
                   help='Whether decoders are shared across datasets')
    p.add_argument('--lr0', default=0.0001, type=float,
                   help='Initial learning rate')
    p.add_argument('--max_grad_norm', default=5.0, type=float,
                   help='Clip gradients to max_grad_norm during training.')
    p.add_argument('--num_train_epochs', default=10, type=int,
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
    p.add_argument('--log_device_placement', action='store_true', default=False,
                   help='Log where compute graph is placed.')
    p.add_argument('--force_gpu_compatible', action='store_true', default=False,
                   help='Throw error if any operations are not GPU-compatible')
    p.add_argument('--label_key', default="label", type=str,
                   help='Key for label field in the batches')
    p.add_argument('--datasets', nargs='+', type=str,
                   help='Key of the dataset(s) to train and evaluate on')
    p.add_argument('--dataset_paths', nargs='+', type=str,
                   help="""Paths to the directory containing the TFRecord files (train.tf, valid.tf, test.tf)
                 for the dataset(s) given by the --datasets flag (in the same order)""")
    p.add_argument('--vocab_path', type=str,
                   help='Path to the shared vocabulary for the datasets')
    p.add_argument('--encoder_architecture', type=str,
                   help='Encoder architecture type (see encoder_factory.py for supported architectures)')
    p.add_argument('--embed_dim', default=128, type=int, help='Dense(hidden) layer size.')
    p.add_argument('--num_filter', default=64, type=int, help='Number of filters for the CNN model.')
    p.add_argument('--max_width', default=5, type=int, help='Maximum window width for the CNN model.')
    p.add_argument('--alphas', nargs='+', type=float, default=[0.5, 0.5],
                   help='alpha for each dataset in the MULT model')
    p.add_argument('--num_layers', type=int, default=2,
                   help='Number off hidden layers of the MLP model.')
    p.add_argument('--model_dirs', nargs='+', type=str, help='Dir to save the best models.')

    return p.parse_args()


def get_num_records(tf_record_filename):
    c = 0
    for record in tf.python_io.tf_record_iterator(tf_record_filename):
        c += 1
    return c


def get_vocab_size(vocab_file_path):
    with open(vocab_file_path, "r") as f:
        line = f.readline().strip()
        vocab_size = int(line)
    return vocab_size


def train_model(model, dataset_info, steps_per_epoch, args):
    dataset_info, model_info = fill_info_dicts(dataset_info, model, args)

    # Get training objective. The inputs are:
    #   1. A dict of { dataset_key: dataset_iterator }
    #
    train_batches = {name: model_info[name]['train_batch'] for name in model_info}
    loss = model.get_multi_task_loss(train_batches)

    # Done building compute graph; set up training ops.

    # Training ops
    global_step_tensor = tf.train.get_or_create_global_step()
    zero_global_step_op = global_step_tensor.assign(0)
    lr = get_learning_rate(args.lr0)
    tvars, grads = get_var_grads(loss)
    train_op = get_train_op(tvars, grads, lr, args.max_grad_norm,
                            global_step_tensor, name='train_op')
    init_ops = [tf.global_variables_initializer(),
                tf.local_variables_initializer()]
    config = get_proto_config(args)
    with tf.train.SingularMonitoredSession(config=config) as sess:
        # Initialize model parameters
        sess.run(init_ops)

        best_eval_acc = dict()
        for dataset_name in model_info:
            _train_init_op = model_info[dataset_name]['train_init_op']
            sess.run(_train_init_op)
            best_eval_acc[dataset_name] = {"epoch": -1,
                                           "acc": float('-inf'),
                                           }

        best_total_acc = float('-inf')
        best_total_acc_epoch = -1

        # Do training
        for epoch in xrange(1, args.num_train_epochs + 1):
            start_time = time()

            total_acc = 0.0

            # Take steps_per_epoch gradient steps
            total_loss = 0
            num_iter = 0
            for _ in tqdm(xrange(steps_per_epoch)):
                step, loss_v, _ = sess.run([global_step_tensor, loss, train_op])
                num_iter += 1
                total_loss += loss_v  # loss_v is sum over a batch from each dataset of the average loss *per training example*
            assert num_iter > 0

            # average loss per batch (which is in turn averaged across examples)
            train_loss = float(total_loss) / float(num_iter)

            # Evaluate held-out accuracy
            if not args.test:  # Validation mode
                # Get performance metrics on each dataset
                for dataset_name in model_info:
                    # _test_batch = dataset_info[dataset_name]['test_dataset'].batch
                    # _pred_op = model.get_predictions(_test_batch, dataset_info[dataset_name]['feature_name'])
                    _pred_op = model_info[dataset_name]['test_pred_op']
                    _eval_labels = model_info[dataset_name]['test_batch'][args.label_key]
                    _eval_iter = model_info[dataset_name]['test_iter']
                    _metrics = compute_held_out_performance(sess, _pred_op,
                                                            _eval_labels,
                                                            _eval_iter, args)
                    model_info[dataset_name]['test_metrics'] = _metrics

                end_time = time()
                elapsed = end_time - start_time
                # Log performance(s)
                str_ = '[epoch=%d/%d step=%d (%d s)] train_loss=%s (per batch)' % (
                    epoch, args.num_train_epochs, np.asscalar(step), elapsed, train_loss)
                for dataset_name in model_info:
                    _num_eval_total = model_info[dataset_name]['test_metrics']['ntotal']
                    _eval_acc = model_info[dataset_name]['test_metrics']['accuracy']
                    _eval_align_acc = model_info[dataset_name]['test_metrics']['aligned_accuracy']
                    str_ += '\n(%s) num_eval_total=%d eval_acc=%f eval_align_acc=%f' % (dataset_name,
                                                                                        _num_eval_total,
                                                                                        _eval_acc,
                                                                                        _eval_align_acc)

                    total_acc += _eval_acc
                    if _eval_acc > best_eval_acc[dataset_name]["acc"]:
                        best_eval_acc[dataset_name]["acc"] = _eval_acc
                        best_eval_acc[dataset_name]["epoch"] = epoch

                        # # save best model
                        # best_ckpt_savers[dataset_name].handle(_eval_acc, sess, global_step_tensor)

                if total_acc > best_total_acc:
                    best_total_acc = total_acc
                    best_total_acc_epoch = epoch

                logging.info(str_)

            else:
                raise NotImplementedError("final evaluation mode not implemented")

        print(best_eval_acc)
        print('Best total accuracy: {} at epoch {}'.format(best_total_acc, best_total_acc_epoch))

        with open('report.txt', 'a') as file:
            for dataset in best_eval_acc.keys():
                file.write(str(dataset))
                file.write(" ")
            file.write("\n")
            for dataset, acc in best_eval_acc.items():
                file.write('Best accuaracy for dataset {}: {}\n'.format(dataset, acc))
            file.write('Best total accuracy: {} at epoch {}\n\n'.format(best_total_acc, best_total_acc_epoch))


def compute_held_out_performance(session, pred_op, eval_label,
                                 eval_iterator, args):
    # pred_op: predicted labels
    # eval_label: gold labels

    # Initializer eval iterator
    session.run(eval_iterator.initializer)

    # Accumulate predictions
    ys = []
    y_hats = []
    while True:
        try:
            y, y_hat = session.run([eval_label, pred_op])
            assert y.shape == y_hat.shape, print(y.shape, y_hat.shape)
            y_list = y.tolist()
            # print("y list type: ", type(y_list))
            # print("y list: ", y_list)
            # y_list = [item for sublist in y_list for item in sublist]
            y_hat_list = y_hat.tolist()
            # y_hat_list = [item for sublist in y_hat_list for item in sublist]
            ys += y_list
            y_hats += y_hat_list
        except tf.errors.OutOfRangeError:
            break

    assert len(ys) == len(y_hats)

    ntotal = len(ys)
    ncorrect = 0
    for i in xrange(len(ys)):
        if ys[i] == y_hats[i]:
            ncorrect += 1
    acc = float(ncorrect) / float(ntotal)

    return {
        'ntotal': ntotal,
        'ncorrect': ncorrect,
        'accuracy': acc,
        'aligned_accuracy': accuracy(ys, y_hats),
    }


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

    # Number of label types in each dataset
    class_sizes = dict()
    for dataset in args.datasets:
        if dataset == 'SSTb':
            class_sizes[dataset] = 5
        else:
            class_sizes[dataset] = 2
    print(class_sizes)

    # Ordering of concatenation for input to decoder
    ordering = dict()
    for order, dataset in enumerate(args.datasets):
        ordering[dataset] = order

    # Read data
    dataset_info = dict()
    for dataset_name in args.datasets:
        dataset_info[dataset_name] = dict()
        # Collect dataset information/statistics
        dataset_info[dataset_name]['feature_name'] = dataset_name  # feature name is just dataset name
        dataset_info[dataset_name]['dir'] = dirs[dataset_name]
        dataset_info[dataset_name]['class_size'] = class_sizes[dataset_name]
        dataset_info[dataset_name]['ordering'] = ordering[dataset_name]
        _dir = dataset_info[dataset_name]['dir']
        # Set paths to TFRecord files
        _dataset_train_path = os.path.join(_dir, "train.tf")
        if args.test:
            _dataset_test_path = os.path.join(_dir, "test.tf")
        else:
            _dataset_test_path = os.path.join(_dir, "valid.tf")
        dataset_info[dataset_name]['train_path'] = _dataset_train_path
        dataset_info[dataset_name]['test_path'] = _dataset_test_path

    vocab_size = get_vocab_size(args.vocab_path)

    class_sizes = {dataset_name: dataset_info[dataset_name]['class_size'] for dataset_name in dataset_info}

    order_dict = {dataset_name: dataset_info[dataset_name]['ordering'] for dataset_name in dataset_info}
    dataset_order = sorted(order_dict, key=order_dict.get)

    # This defines ALL the features that ANY model possibly needs access
    # to. That is, some models will only need a subset of these features.
    FEATURES = {
        'label': tf.FixedLenFeature([], dtype=tf.int64),
        'tokens': tf.VarLenFeature(dtype=tf.int64),
        'tokens_length': tf.FixedLenFeature([], dtype=tf.int64),
        # 'types': tf.VarLenFeature(dtype=tf.int64),
        # 'type_counts': tf.VarLenFeature(dtype=tf.int64),
        # 'types_length': tf.FixedLenFeature([], dtype=tf.int64),
        # 'bow': tf.FixedLenFeature([vocab_size], dtype=tf.float32),
    }

    logging.info("Creating computation graph...")
    with tf.Graph().as_default():

        # Creating the batch input pipelines.  These will load & batch
        # examples from serialized TF record files.
        for dataset_name in dataset_info:
            _train_path = dataset_info[dataset_name]['train_path']
            ds = build_input_dataset(_train_path, FEATURES, args.batch_size, is_training=True)
            dataset_info[dataset_name]['train_dataset'] = ds

            # Validation or test dataset
            _test_path = dataset_info[dataset_name]['test_path']
            ds = build_input_dataset(_test_path, FEATURES, args.eval_batch_size, is_training=False)
            dataset_info[dataset_name]['test_dataset'] = ds

        # This finds the size of the largest training dataset.
        training_files = [dataset_info[dataset_name]['train_path'] for dataset_name in dataset_info]
        max_N_train = max([get_num_records(tf_rec_file) for tf_rec_file in training_files])

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
        #        args: test batch (with <batch_len> examples), name of feature to predict
        #        returns: Tensor of size <batch_len> specifying the predicted label
        #                 for the specified feature for each of the examples in the batch
        # Seth's multi-task VAE model
        # if args.model == 'mlvae':
        #   m = MultiLabel(class_sizes=class_sizes,
        #                  dataset_order=dataset_order,
        #                  encoders=encoders,
        #                  decoders=decoders,
        #                  hp=None)
        # Felicity's discriminative baseline

        encoder_hp = None
        encoders = build_encoders(vocab_size, args, encoder_hp)
        hp = set_hp(args)

        if args.model == 'mult':
            m = Mult(class_sizes=class_sizes,
                     dataset_order=dataset_order,
                     encoders=encoders,
                     hp=hp,
                     is_training=True)
        else:
            raise ValueError("unrecognized model: %s" % args.model)

        # Do training
        train_model(m, dataset_info, steps_per_epoch, args)


def set_hp(args):
    # TODO get hyperparameters from arguments
    return HParams(embed_dim=args.embed_dim,
                   num_filter=args.num_filter,
                   max_width=args.max_width,
                   word_embed_dim=args.word_embed_dim,
                   alphas=args.alphas,
                   labels_key="label",
                   inputs_key="tokens",
                   token_lengths_key="tokens_length",
                   l2_weight=0.0,
                   dropout_rate=0.5,
                   num_layers=args.num_layers)


def get_learning_rate(learning_rate):
    return tf.constant(learning_rate)


def get_train_op(tvars, grads, learning_rate, max_grad_norm, step,
                 name=None):
    with tf.name_scope(name):
        opt = tf.train.AdamOptimizer(learning_rate,
                                     epsilon=1e-6,
                                     beta1=0.85,
                                     beta2=0.997)
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


def fill_info_dicts(dataset_info, model, args):
    # Organizes inputs to the computation graph
    # The corresponding outputs are defined in train_model()

    # dataset_info: dict containing dataset-specific parameters/statistics
    #  e.g., number of classes, path to data
    # model_info: dict containing dataset-specific information w.r.t. running the model
    #  e.g., data iterators, batches, prediction operations

    # Storage for pointers to dataset-specific Tensors
    model_info = dict()
    for dataset_name in dataset_info:
        model_info[dataset_name] = dict()

    # Data iterators, etc.
    for dataset_name in dataset_info:
        # Training data, iterator, and batch
        _train_dataset = dataset_info[dataset_name]['train_dataset']
        _train_iter = _train_dataset.iterator
        _train_init_op = _train_dataset.init_op
        _train_batch = _train_dataset.batch
        model_info[dataset_name]['train_iter'] = _train_iter
        model_info[dataset_name]['train_init_op'] = _train_init_op
        model_info[dataset_name]['train_batch'] = _train_batch

        # Held-out test data, iterator, batch, and prediction operation
        _test_dataset = dataset_info[dataset_name]['test_dataset']
        _test_iter = _test_dataset.iterator
        _test_batch = _test_dataset.batch
        _test_pred_op = model.get_predictions(_test_batch, dataset_info[dataset_name]['feature_name'])
        model_info[dataset_name]['test_iter'] = _test_iter
        model_info[dataset_name]['test_batch'] = _test_batch
        model_info[dataset_name]['test_pred_op'] = _test_pred_op
        if args.test:
            logging.info("Using test data for evaluation.")
        else:
            logging.info("Using validation data for evaluation.")

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
    for dataset_name in model_info:
        model_info[dataset_name]['feature_dict'] = _create_feature_dict(dataset_name, dataset_info, model_info)

    # Return dataset_info dict and model_info dict
    return dataset_info, model_info


def build_input_dataset(tfrecord_path, batch_features, batch_size, is_training=True):
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

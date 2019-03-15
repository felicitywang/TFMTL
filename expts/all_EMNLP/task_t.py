# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

import argparse
import datetime
import json
import os
import threading

import model
import numpy as np
import tensorflow as tf
from six.moves import xrange

from mtl.util.metrics import metric2func

# from gpu_util import setup_one_gpu
# setup_one_gpu()

tf.logging.set_verbosity(tf.logging.INFO)

TASK_TO_METRIC = {
    'FNC-1': 'Acc',
    'Stance': 'F1_PosNeg_Macro',
    'MultiNLI': 'Acc',
    'Topic5': 'MAE_Macro',
    'Topic2': 'Recall_Macro',
    'Target': 'F1_Macro',
    'ABSA-R': 'Acc',
    'ABSA-L': 'Acc',

    'SST2': 'Acc'
}


class EvaluationRunHook(tf.train.SessionRunHook):
    """EvaluationRunHook performs continuous evaluation of the model.

    Args:
      checkpoint_dir (string): Dir to store model checkpoints
      fetches (dict): Dictionary of tf.Tensor ops
      graph (tf.Graph): Evaluation graph
      eval_frequency (int): Frequency of evaluation every n train checkpoints
    """

    def __init__(self,
                 checkpoint_dir,
                 fetches,
                 graph,
                 test_mode,
                 eval_frequency,
                 eval_corpus,
                 **kwargs):

        self._checkpoint_dir = checkpoint_dir

        tf.logging.info("Checkpoint directory: %s", checkpoint_dir)
        tf.logging.info("Evaluation frequency: %s steps", eval_frequency)

        self._kwargs = kwargs
        self._eval_corpus = eval_corpus
        self._eval_every = eval_frequency
        self._latest_checkpoint = None
        self._checkpoints_since_eval = 0
        self._graph = graph
        self._test_mode = test_mode

        self._mean_error = float('inf')
        self._best_acc = 0.0
        self._best_iter = 0
        self._best_score = 0.0
        self._best_score_iter = 0

        # Fetches to evaluate
        if "seq1" in fetches:
            self._seq1_op = fetches['seq1']
        else:
            self._seq1_op = None
        self._y_op = fetches['y']
        self._y_hat_op = fetches['y_hat']
        self._class_error_op = fetches['class_error']

        # With the graph object as default graph
        # See https://www.tensorflow.org/api_docs/python/tf/Graph#as_default
        # Adds ops to the graph object
        with graph.as_default():
            # Creates a global step to contain a counter for the global
            # training step.
            self._gs = global_step = tf.train.get_or_create_global_step()

            # Saver class add ops to save and restore variables to and from
            # checkpoint.
            self._saver = tf.train.Saver(
                tf.trainable_variables() + [global_step])

        # MonitoredTrainingSession runs hooks in background threads and it
        # doesn't wait for the thread from the last session.run() call to
        # terminate to invoke the next hook, hence locks.
        self._eval_lock = threading.Lock()
        self._checkpoint_lock = threading.Lock()

        self._file_writer = tf.summary.FileWriter(
            os.path.join(checkpoint_dir, 'eval'), graph=graph)
        # self._file_writer = tf.summary.FileWriter(
        #   checkpoint_dir, graph=graph)

    def after_run(self, run_context, run_values):
        # Always check for new checkpoints in case a single evaluation
        # takes longer than checkpoint frequency and _eval_every is >1
        self._update_latest_checkpoint()

        if self._eval_lock.acquire(False):
            try:
                if self._checkpoints_since_eval >= self._eval_every:
                    self._checkpoints_since_eval = 0
                    self._run_eval()
            finally:
                self._eval_lock.release()

    def _update_latest_checkpoint(self):
        """Update the latest checkpoint file created in the output dir."""
        if self._checkpoint_lock.acquire(False):
            try:
                latest = tf.train.latest_checkpoint(self._checkpoint_dir)
                if not latest == self._latest_checkpoint:
                    self._checkpoints_since_eval += 1
                    self._latest_checkpoint = latest
            finally:
                self._checkpoint_lock.release()

    def end(self, session):
        """Called at then end of session to make sure we always evaluate."""
        self._update_latest_checkpoint()

        with self._eval_lock:
            self._run_eval()

    def _run_eval(self):
        #    tf.logging.info("Running model evaluation & generating summaries.")
        # config = tf.ConfigProto(device_count = {'GPU': 0})
        config = tf.ConfigProto(inter_op_parallelism_threads=4,
                                intra_op_parallelism_threads=4)
        # config=None
        with tf.Session(graph=self._graph, config=config) as session:
            session.run([
                tf.tables_initializer(),
                tf.local_variables_initializer(),
                tf.global_variables_initializer()
            ])

            # TODO: swap with above?
            self._saver.restore(session, self._latest_checkpoint)

            # Retrieve global step
            train_step = session.run(self._gs)

            # Run the evaluation
            #      tf.logging.info('Starting evaluation for step: %d', train_step)
            ts = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
            eval_step = 0
            ys = []
            yhats = []
            topic_to_id = {}
            topics = []
            total_error = 0.0
            while True:
                try:
                    error, seq1, y, y_hat = session.run([self._class_error_op,
                                                         self._seq1_op,
                                                         self._y_op,
                                                         self._y_hat_op])
                    # if self._seq1_op is not None:
                    #   seq1 = session.run(self._seq1_op)
                    # else:
                    #   seq1 = None

                    try:
                        # print(len(seq1.tolist()), len(y.tolist()), len(y_hat.tolist()))
                        # Collect seq1 topics if possible
                        for seq in seq1.tolist():
                            seq_tuple = tuple(seq)
                            if seq_tuple not in topic_to_id:
                                topic_to_id[seq_tuple] = len(topic_to_id)
                            tid = topic_to_id[seq_tuple]
                            topics.append(tid)
                    except:
                        if seq1 is not None:
                            raise
                        else:
                            pass

                    total_error += error
                    ys += y.tolist()
                    yhats += y_hat.tolist()

                    if eval_step % 100 == 0 and eval_step > 0:
                        tf.logging.info("On evaluation step: %d", eval_step)
                    eval_step += 1
                except tf.errors.OutOfRangeError:
                    tf.logging.info("Reached end of evaluation dataset.")
                    break

            assert eval_step > 0, "ERROR: didn't complete any evaluation steps"

            # Task-specific loss
            self._mean_error = mean_error = total_error / float(eval_step)
            assert np.isfinite(mean_error), "Non-finite error {}".format(
                mean_error)
            assert not np.isnan(mean_error), "NaN error {}".format(mean_error)

            # Number of evaluation steps
            # tf.logging.info("Completed %d evaluation steps.", eval_step)
            assert len(ys) == len(yhats)
            ntotal = len(ys)
            ncorrect = len([True for x, y in zip(ys, yhats) if x == y])
            acc = float(ncorrect) / float(len(ys))
            summary = tf.Summary(value=[tf.Summary.Value(tag="heldout_accuracy",
                                                         simple_value=acc)])
            self._file_writer.add_summary(summary, global_step=train_step)
            self._file_writer.add_summary(
                tf.Summary(value=[tf.Summary.Value(tag="heldout_error",
                                                   simple_value=mean_error)]),
                global_step=train_step)
            self._file_writer.flush()

            if acc > self._best_acc:
                self._best_acc = acc
                self._best_iter = train_step

            # Task-specific metric
            corpus_config = model.get_corpus_config(self._eval_corpus)
            labels = corpus_config['labels']
            assert type(self._eval_corpus) is str
            parts = os.path.split(self._eval_corpus)
            task = parts[-1]
            assert task in TASK_TO_METRIC
            metric = TASK_TO_METRIC[task]
            metric_fn = metric2func(metric)

            if task in ['Topic2', 'Topic5']:
                # tf.logging.info("TOPICS, YS")
                # tf.logging.info(topics)
                # tf.logging.info(ys)
                # tf.logging.info("LENGTHS TOPICS, YS, YHATS")
                # tf.logging.info(len(topics))
                # tf.logging.info(len(ys))
                # tf.logging.info(len(yhats))
                assert len(topics) == len(ys)
                assert len(topics) == len(yhats)
                tf.logging.info(
                    "Found %d topics for evaluation." % len(topic_to_id))
                pass
            else:
                topics = [0] * len(ys)

            score = metric_fn(ys, yhats, labels, topics)

            if score > self._best_score:
                self._best_score = score
                self._best_score_iter = train_step

            # Log results
            tf.logging.info(
                "[step=%s] [%s test=%s] Error: %.4f  Acc: %.4f (%d/%d)  BestAcc[%d]: %.4f  %s: %.4f (Best%s[%d]: %.4f)" % (
                    train_step,
                    task,
                    self._test_mode,
                    mean_error,
                    acc,
                    ncorrect,
                    ntotal,
                    self._best_iter,
                    self._best_acc,
                    metric,
                    score,
                    metric,
                    self._best_score_iter,
                    self._best_score))


def run(target,
        cluster_spec,
        is_chief,
        hyperparams,
        job_dir,
        reuse_job_dir,
        eval_frequency,
        save_secs,
        # save_steps,
        train_corpora,
        eval_corpus):
    # Alias
    hp = hyperparams

    # If job_dir_reuse is False then remove the job_dir if it exists
    if not reuse_job_dir:
        if tf.gfile.Exists(job_dir):
            tf.gfile.DeleteRecursively(job_dir)
            tf.logging.info(
                "Deleted job_dir {} to avoid re-use".format(job_dir))
        else:
            tf.logging.info("No job_dir available to delete")
    else:
        tf.logging.info("Reusing job_dir {} if it exists".format(job_dir))

    # If the server is chief which is `master` In between graph
    # replication Chief is one node in the cluster with extra
    # responsibility and by default is worker task zero. We have
    # assigned master as the chief.
    #
    # See https://youtu.be/la_M6bCV91M?t=1203 for details on distributed
    # TensorFlow and motivation about chief.
    if is_chief:
        tf.logging.info("is_chief==True; creating evaluation graph.")
        evaluation_graph = tf.Graph()
        with evaluation_graph.as_default():
            # Seed TensorFlow RNG
            tf.logging.info(
                "Setting TF random seed (eval): {}".format(hp.random_seed))
            tf.set_random_seed(hp.random_seed)

            # if hp.TEST == 'yes':
            #   user_choice = input("TEST MODE: Evaluating on TEST data. Are you sure? (y/N):\n")
            #   if user_choice in('y', 'Y'):
            #     pass
            #   else:
            #     import sys
            #     sys.exit(0)
            # elif hp.TEST == 'no':
            #   pass
            # else:
            #   raise ValueError(hp.TEST)

            eval_batch = model.input_fn(eval_corpus,
                                        model.EVAL,
                                        TEST_MODE=hp.TEST == 'yes',
                                        num_epochs=1,
                                        batch_size=hp.eval_batch_size,
                                        shuffle=False,
                                        one_shot=True,
                                        num_input_seq=hp.num_input_seq)
            fetches = model.model_fn(eval_corpus, model.EVAL, eval_batch, hp)

            if hp.TEST == 'yes':
                tf.logging.info("Including dev batch in evaluation")
                dev_batch = model.input_fn(eval_corpus,
                                           model.EVAL,
                                           TEST_MODE=False,
                                           num_epochs=1,
                                           batch_size=hp.eval_batch_size,
                                           shuffle=False,
                                           one_shot=True,
                                           num_input_seq=hp.num_input_seq)
                dev_fetches = model.model_fn(eval_corpus, model.EVAL, dev_batch,
                                             hp)

        hooks = [EvaluationRunHook(
            job_dir,
            fetches,
            evaluation_graph,
            hp.TEST == 'yes',
            eval_frequency,
            eval_corpus
        )]

        if hp.TEST == 'yes':
            hooks.append(EvaluationRunHook(
                job_dir,
                dev_fetches,
                evaluation_graph,
                False,
                eval_frequency,
                eval_corpus
            ))


    else:
        tf.logging.info("is_chief==False; no hooks")
        hooks = []

    # Create a new graph and specify that as default
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
            # Seed TensorFlow RNG
            tf.logging.info(
                "Setting TF random seed (training): {}".format(hp.random_seed))
            tf.set_random_seed(hp.random_seed)

            # Features and label tensors as read using filename queue
            # tf.logging.info("%d train epochs" % (hp.num_epochs))
            assert type(train_corpora) is list

            if hp.semisup == 'eval':
                num_corpus = len(train_corpora) + 1
            elif hp.semisup == 'transductive':
                num_corpus = len(train_corpora) * 2
            elif hp.semisup == 'dev-only':
                num_corpus = len(train_corpora) * 2
            elif hp.semisup == 'all':
                num_corpus = len(train_corpora) * 3
            elif hp.semisup == 'none':
                num_corpus = len(train_corpora)
            else:
                raise ValueError(hp.semisup)

            batch_size_per_corpus = int(hp.train_batch_size / num_corpus)
            tf.logging.info("Batch size per corpus: %d" % batch_size_per_corpus)

            assert batch_size_per_corpus > 1

            batches = []
            corpora = []
            for corpus in train_corpora:
                batch = model.input_fn(corpus,
                                       model.TRAIN,
                                       num_epochs=None,
                                       shuffle=True,
                                       batch_size=batch_size_per_corpus,
                                       num_input_seq=hp.num_input_seq)

                # Labeled training corpus
                batches.append(batch)
                corpora.append(corpus)

                if (hp.semisup == 'eval' and len(
                    batches) == 1) or hp.semisup == 'transductive' or hp.semisup == 'all':
                    tf.logging.info(
                        "Eval data as unlabeled training data for target task")

                    # if hp.TEST == 'yes':
                    #   user_choice = input("TRAINING ON TEST. Are you sure? (y/N):\n")
                    #   if user_choice in('y', 'Y'):
                    #     pass
                    #   else:
                    #     import sys
                    #     sys.exit(0)
                    # elif hp.TEST == 'no':
                    #   pass
                    # else:
                    #   raise ValueError(hp.TEST)

                    # Unlabeled validation or test corpus
                    batch = model.input_fn(corpus,
                                           model.EVAL,
                                           num_epochs=None,
                                           shuffle=True,
                                           TEST_MODE=hp.TEST == 'yes',
                                           batch_size=batch_size_per_corpus,
                                           num_input_seq=hp.num_input_seq)

                    batches.append(batch)
                    corpora.append(corpus)

                if hp.semisup == 'dev-only' or hp.semisup == 'all':
                    tf.logging.info(
                        "Development data as unlabeled training data for target task")

                    # Unlabeled validation corpus
                    assert hp.TEST == 'yes'  # only in addition to test corpus
                    batch = model.input_fn(corpus,
                                           model.EVAL,
                                           num_epochs=None,
                                           shuffle=True,
                                           TEST_MODE=False,
                                           # EVAL + False = dev
                                           batch_size=batch_size_per_corpus,
                                           num_input_seq=hp.num_input_seq)

                    batches.append(batch)
                    corpora.append(corpus)

            assert len(batches) == len(corpora)

            # Returns the training graph and global step tensor
            train_op, loss, global_step_tensor = model.model_fn(corpora,
                                                                model.TRAIN,
                                                                batches,
                                                                hp)

            merged_summary_op = tf.summary.merge_all()

            total_parameters = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                tf.logging.info("%d: %s" % (variable_parameters, variable))
                total_parameters += variable_parameters
            num_byte = total_parameters * 4
            tf.logging.info("Model size: %d Mb (%d parameters)",
                            num_byte / 1000000, total_parameters)

        tf.logging.info("Creating MonitoredTrainingSession for training.")
        tf.logging.info("Saving checkpoints every %d seconds." % save_secs)
        # tf.logging.info("Saving checkpoints every %d steps." % save_steps)
        if hp.device_type == 'CPU':
            config = tf.ConfigProto(inter_op_parallelism_threads=4,
                                    intra_op_parallelism_threads=4)
            # config = None
        elif hp.device_type == 'GPU':
            config = None
        else:
            raise ValueError(
                "Unexpected device type: {}".format(hp.device_type))
        with tf.train.MonitoredTrainingSession(config=config,
                                               master=target,
                                               is_chief=is_chief,
                                               checkpoint_dir=job_dir,
                                               hooks=hooks,
                                               save_checkpoint_secs=save_secs,
                                               # save_checkpoint_steps=save_steps,
                                               log_step_count_steps=50,
                                               save_summaries_steps=50) as sess:
            # Run the training graph which returns the step number as
            # tracked by the global step tensor.  When train epochs is
            # reached, session.should_stop() will be true.
            for train_step in xrange(hp.num_train_steps):
                step, _, _, loss_val = sess.run([global_step_tensor,
                                                 train_op,
                                                 merged_summary_op,
                                                 loss])
                tf.logging.info("Training loss: {}".format(loss_val))


def dispatch(hp, *args, **kwargs):
    """Parse TF_CONFIG to cluster_spec and call run() method TF_CONFIG
    environment variable is available when running using gcloud either
    locally or on cloud. It has all the information required to create a
    ClusterSpec which is important for running distributed code.

    """

    tf_config = os.environ.get('TF_CONFIG')

    # If TF_CONFIG is not available run local
    if not tf_config:
        tf.logging.info("TF_CONFIG is not available.")
        return run(target='', cluster_spec=None, is_chief=True,
                   hyperparams=hp, *args,
                   **kwargs)

    assert not hp.clsp, "Trying to run non-local"

    tf_config_json = json.loads(tf_config)

    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    if job_name is None or task_index is None:
        tf.logging.info("Cluster information is empty; running local.")
        return run(target='', cluster_spec=None, is_chief=True,
                   hyperparams=hp, *args,
                   **kwargs)

    cluster_spec = tf.train.ClusterSpec(cluster)
    server = tf.train.Server(cluster_spec, job_name=job_name,
                             task_index=task_index)

    # Wait for incoming connections forever Worker ships the graph to
    # the ps server The ps server manages the parameters of the model.
    #
    # See a detailed video on distributed TensorFlow
    # https://www.youtube.com/watch?v=la_M6bCV91M
    if job_name == 'ps':
        server.join()
        return
    elif job_name in ['master', 'worker']:
        return run(server.target, cluster_spec, is_chief=(job_name == 'master'),
                   hyperparams=hp, *args, **kwargs)
    else:
        raise ValueError('unrecognized job name: %s' % (job_name))


if __name__ == "__main__":
    # Retain a reserved GPU
    config = tf.ConfigProto(inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    # config = None
    sess = tf.Session(config=config)
    dummy = tf.constant(1.0)
    sess.run(dummy)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-corpora',
                        required=True,
                        nargs='+',
                        type=str,
                        help="""\
                      Training files local or GCS. The path names must have
                      corresponding JSON configuration files with the same
                      root path name(s).
                      """)
    parser.add_argument('--eval-corpus',
                        required=True,
                        type=str,
                        help="""\
                      Evaluation file local or GCS. The path name must have
                      a corresponding JSON configuration file with the same
                      root path name.
                      """)
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help="""\
                      GCS or local dir for checkpoints, exports, and
                      summaries. Use an existing directory to load a
                      trained model, or a new directory to retrain""")
    parser.add_argument('--reuse-job-dir',
                        action='store_true',
                        default=False,
                        help="""\
                      Flag to decide if the model checkpoint should
                      be re-used from the job-dir. If False then the
                      job-dir will be deleted
                      """)
    parser.add_argument('--eval-frequency',
                        type=int,
                        default=1,
                        help='Perform one evaluation per N checkpoints.')
    parser.add_argument('--save-secs',
                        type=int,
                        default=30,
                        help='Seconds between saving checkpoints.')
    # parser.add_argument('--save-steps',
    #                    type=int,
    #                    default=50,
    #                    help='Steps between saving checkpoints.')
    parser.add_argument('--hparams',
                        type=str,
                        help='Comma separated list of "name=value" pairs.')
    parser.add_argument('--hparams-file',
                        type=str,
                        help='Path to hyper-parameter file')
    parser.add_argument('--device-type',
                        type=str,
                        choices=['CPU', 'GPU'],
                        help='Kind of device')
    parser.add_argument('--verbosity',
                        choices=[
                            'DEBUG',
                            'ERROR',
                            'FATAL',
                            'INFO',
                            'WARN'
                        ],
                        default='INFO',
                        help='Set logging verbosity')
    args, unknown = parser.parse_known_args()

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)

    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[args.verbosity] / 10)
    del args.verbosity

    if unknown:
        tf.logging.warn('Unknown arguments: {}'.format(unknown))

    # Default hyper-parameters
    hp = tf.contrib.training.HParams(
        num_train_steps=2000,
        learning_rate=1e-3,
        #    learning_rate = 1e-4,
        #    lr_schedule = 'cosine_decay_restarts',
        lr_schedule='constant',
        max_grad_norm=10.0,
        #    max_grad_norm = 10.0,  # no clipping
        num_hidden_layer=1,  # size of task-specific MLP
        hidden_dim=512,  # size of task-specific MLP
        train_batch_size=128,
        eval_batch_size=256,
        keep_prob=1.0,
        optimizer='adafactor',
        #    optimizer = 'adam',
        #    optimizer = 'sgd',
        embed_dim=256,  # word embedding size
        #    max_embed_norm = 1.0,
        #    encoder_inputs = 'x+task',
        encoder_inputs='x',

        encoder='generic_serial_encoder',
        encoder_hparams='symmetric_serial_birnn',

        #    encoder = 'generic_encoder',
        #    encoder_hparams = 'single_birnn_default',

        #    encoder_hparams = 'serial_cnn_default_symmetric',
        #    encoder_hparams = 'serial_cnn_default_symmetric_regularized',
        # encoder_hparams = 'serial_birnn',

        # encoder_hparams = 'serial_cnn_larger',
        #    encoder_hparams = 'serial_cnn',
        #    prior_encoder = 'convpool',
        #    prior_encoder_hparams = 'convpool_small',
        # semisup = 'none',  # none|eval|transductive|dev-only
        # semisup = 'transductive',
        decoder_sharing='tied',
        TEST='yes',
        #    decoder = 'tcn',
        #    decoder_hparams ='tcn_small',
        model='gmtl',
        #    z_posterior = 'hf', # 'diag_gaussian',  # hf
        #    z_posterior = 'diag_gaussian',
        #    flow_depth = 4,
        #    label_prior = 'empirical',
        #    latent_dim = 64,
        #    model = 'gmtl',
        decoder='resnet',
        decoder_hparams='resnet_default',
        # decoder_hparams = 'resnet_single_layer_wide',
        # decoder_hparams = 'resnet_single_layer',
        decoder_cond='code+task+label',
        #    decoder_cond = 'task+label',
        #    decoder_cond = 'task+code',
        #    decoder_cond = 'code',
        decoder_error='sum',  # sum or avg
        label_embed_dim=64,
        tau=0.5,  # (0, \infty): Concrete distribution temperature
        #    nll_anneal_from = 0.01,
        #    nll_anneal_steps = 10000,
        # alpha_scale = 0.01,
        beta=1.0,
        #    beta = 0.01,  # weight on generative loss (0=disc only)
        # beta_0 = 0.5,  # set to 0. for discriminative-only
        # beta_min = 0.5,
        # beta_decay_steps = 500,
        # beta_decay_rate = 0.5,
        embed_init_stddev=1e-3,
        code_keep_prob=1.0,  # dropout feature code?,
        # gamma = 0.5,  # set to 1.0 for single-task objective

        # TODO
        ablate_gen_loss=False,  # True: disc only; False: disc+gen
        semisup='eval',  # none|eval|transductive|dev-only
        num_input_seq=2,

        even_task_weighting=True,
        clsp=True,
        random_seed=1066
    )

    # Update from command-line arguments; comma-separated list of
    # hyper-parameters.
    if args.hparams:
        tf.logging.info('Updating hyper-parameters from command-line.')
        hp.parse(args.hparams)
    del args.hparams

    if args.hparams_file:
        tf.logging.info(
            'Updating hyper-parameters from: %s.' % args.hparams_file)
        hp.read_json(args.hparams_file)
    del args.hparams_file

    # Write hyper-parameters
    path = args.job_dir + '.json'
    tf.logging.info('Writing hyper-parameters to: %s.' % path)
    with tf.gfile.GFile(path, 'w') as f:
        json_string = hp.to_json(indent=2)
        tf.logging.info('Hyper-parameters:')
        tf.logging.info(json_string)
        f.write(json_string)

    # Check assumptions
    assert args.train_corpora[0] == args.eval_corpus

    # Add the number of tasks
    tf.logging.info("%d tasks", len(args.train_corpora))
    hp.add_hparam("num_tasks", len(args.train_corpora))

    hp.add_hparam("device_type", args.device_type)
    del args.device_type

    # Logging the version.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("TensorFlow version: %s", tf.__version__)

    # Seed numpy RNG
    np.random.seed(hp.random_seed)

    dispatch(hp, **args.__dict__)

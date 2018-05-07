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
import json
import os
import threading
import datetime

import model
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import variables
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.saved_model import signature_constants as sig_constants

tf.logging.set_verbosity(tf.logging.INFO)


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
               eval_frequency,
               **kwargs):

    self._checkpoint_dir = checkpoint_dir

    tf.logging.info("Checkpoint directory: %s", checkpoint_dir)
    tf.logging.info("Evaluation frequency: %s steps", eval_frequency)

    self._kwargs = kwargs
    self._eval_every = eval_frequency
    self._latest_checkpoint = None
    self._checkpoints_since_eval = 0
    self._graph = graph

    # Fetches to evaluate
    self._loss_op = fetches['loss']
    self._len_op = fetches['length']

    # With the graph object as default graph
    # See https://www.tensorflow.org/api_docs/python/tf/Graph#as_default
    # Adds ops to the graph object
    with graph.as_default():
      # Creates a global step to contain a counter for the global
      # training step.
      self._gs = global_step = tf.train.get_or_create_global_step()

      # Saver class add ops to save and restore variables to and from
      # checkpoint.
      self._saver = tf.train.Saver(tf.trainable_variables() + [global_step])

    # MonitoredTrainingSession runs hooks in background threads and it
    # doesn't wait for the thread from the last session.run() call to
    # terminate to invoke the next hook, hence locks.
    self._eval_lock = threading.Lock()
    self._checkpoint_lock = threading.Lock()
    self._file_writer = tf.summary.FileWriter(
      os.path.join(checkpoint_dir, 'eval'), graph=graph)

  def after_run(self, run_context, run_values):
    # Always check for new checkpoints in case a single evaluation
    # takes longer than checkpoint frequency and _eval_every is >1
    self._update_latest_checkpoint()

    if self._eval_lock.acquire(False):
      try:
        if self._checkpoints_since_eval > self._eval_every:
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
    tf.logging.info("Running model evaluation & generating summaries.")
    config = tf.ConfigProto(device_count = {'GPU': 0})
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
      total_loss = 0.0

      # Run the evaluation
      tf.logging.info('Starting evaluation for step: %d', train_step)
      ts = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
      eval_step = 0
      while True:
        try:
          loss_v, len_v = session.run([self._loss_op, self._len_op])
          total_loss += loss_v / len_v
          if eval_step % 100 == 0:
            tf.logging.info("On evaluation step: %d", eval_step)
          eval_step += 1
        except tf.errors.OutOfRangeError:
          tf.logging.info("Reached end of evaluation dataset.")
          break

      assert eval_step > 0, "ERROR: didn't complete any evaluation steps"

      # Number of evaluation steps
      tf.logging.info("Completed %d evaluation steps.", eval_step)

      total_loss = total_loss / eval_step
      summary = tf.Summary(value=[tf.Summary.Value(tag="eval_loss",
                                                   simple_value=total_loss)])
      self._file_writer.add_summary(summary, global_step=train_step)
      self._file_writer.flush()

      # Log results
      tf.logging.info("[%s] Loss: %f Perplexity: %d", ts, total_loss,
                      np.exp(total_loss))


def run(target,
        cluster_spec,
        is_chief,
        hyperparams,
        job_dir,
        reuse_job_dir,
        eval_frequency,
        save_secs,
        train_file,
        eval_file):
  # Alias
  hp = hyperparams

  # If job_dir_reuse is False then remove the job_dir if it exists
  if not reuse_job_dir:
    if tf.gfile.Exists(job_dir):
      tf.gfile.DeleteRecursively(job_dir)
      tf.logging.info("Deleted job_dir {} to avoid re-use".format(job_dir))
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
      eval_batch = model.input_fn(eval_file,
                                  num_epochs=1,
                                  batch_size=hp.eval_batch_size,
                                  shuffle=False, one_shot=True)
      total_len = tf.reduce_sum(eval_batch[model.LEN_FIELD])
      fetches = model.model_fn(model.EVAL, eval_batch, hp)

    hooks = [EvaluationRunHook(
      job_dir,
      fetches,
      evaluation_graph,
      eval_frequency,
    )]
  else:
    tf.logging.info("is_chief==False; no hooks")
    hooks = []

  # Create a new graph and specify that as default
  with tf.Graph().as_default():
    # Placement of ops on devices using replica device setter which
    # automatically places the parameters on the `ps` server and the
    # `ops` on the workers
    #
    # See:
    # https://www.tensorflow.org/api_docs/python/tf/train/replica_device_setter
    with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):

      # Features and label tensors as read using filename queue
      tf.logging.info("%d train epochs" % (hp.num_epochs))
      batch = model.input_fn(train_file,
                             num_epochs=hp.num_epochs, shuffle=False,
                             batch_size=hp.train_batch_size)

      # Returns the training graph and global step tensor
      train_op, global_step_tensor = model.model_fn(model.TRAIN,
                                                    batch, hp)

      merged_summary_op = tf.summary.merge_all()

      total_parameters = 0
      for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
          variable_parameters *= dim.value
          total_parameters += variable_parameters
      num_byte = total_parameters * 4
      tf.logging.info("Model size: %d Mb", num_byte / 1000000)

    tf.logging.info("Creating MonitoredTrainingSession for training.")
    tf.logging.info("Saving checkpoints every %d seconds." % save_secs)
    with tf.train.MonitoredTrainingSession(master=target,
                                           is_chief=is_chief,
                                           checkpoint_dir=job_dir,
                                           hooks=hooks,
                                           save_checkpoint_secs=save_secs,
                                           log_step_count_steps=1000,
                                           save_summaries_steps=1000) as sess:
      # Global step to keep track of global number of steps particularly in
      # distributed setting
      step = global_step_tensor.eval(session=sess)

      # Run the training graph which returns the step number as
      # tracked by the global step tensor.  When train epochs is
      # reached, session.should_stop() will be true.
      while not sess.should_stop():
        step, _, _ = sess.run([global_step_tensor, train_op, merged_summary_op])


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
  parser = argparse.ArgumentParser()
  parser.add_argument('--train-file',
                      required=True,
                      type=str,
                      help='Training file local or GCS')
  parser.add_argument('--eval-file',
                      required=True,
                      type=str,
                      help='Evaluation file local or GCS')
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
  parser.add_argument('--hparams',
                      type=str,
                      help='Comma separated list of "name=value" pairs.')
  parser.add_argument('--hparams-file',
                      type=str,
                      help='Path to hyper-parameter file')
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
    num_epochs = 50,
    learning_rate = 0.0001,
    max_grad_norm = 0,
    train_batch_size = 64,
    eval_batch_size = 256,
    keep_prob = 1.0,
    optimizer = 'adafactor',
    embed_dim = 512,
    decoder = 'rnn',
    decoder_hparams = 'rnn_default'
    #decoder = 'resnet',
    #decoder_hparams = 'resnet_large'
  )

  # Update from command-line arguments; comma-separated list of
  # hyper-parameters.
  if args.hparams:
    tf.logging.info('Updating hyper-parameters from command-line.')
    hp.parse(args.hparams)
  del args.hparams

  if args.hparams_file:
    tf.logging.info('Updating hyper-parameters from: %s.' % args.hparams_file)
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

  # Logging the version.
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info("TensorFlow version: %s", tf.__version__)

  dispatch(hp, **args.__dict__)

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
# See the License for the specific lang governing permissions and
# limitations under the License.
# =============================================================================
"""Generate running scripts and qsub on the grid

TODO
Usage:
  1. Modify config files
  2. Run: python qsub_stl_jobs.py config_json_name encod_json_name
"""

import collections
import itertools
import json
import os
import re
import subprocess
import sys
import time
from functools import reduce

from tqdm import tqdm

from mtl.util.util import load_json

NUM_ARGS = 2
JOBNUM_IDX = 0
NUMSLOTS_IDX = -1

DEFAULT_EMAIL = ''
DEFAULT_EMAIL_PREFS = 'n'

# arguments that are of type list
list_arguments = ['activation_fns']


class MetaConfig(object):
  def __init__(self, config):
    try:
      self.debug = config.pop('debug')
      self.bashrc_path = config.pop('bashrc_path')
      self.cpu_venv = config.pop('cpu_venv')
      self.gpu_venv = config.pop('gpu_venv')
      self.cpu_or_gpu = config.pop('cpu_or_gpu')
      self.code_path = config.pop('code_path')
      self.root_dir = config.pop('root_dir')
      self.results_dir = config.pop('results_dir')
      self.email = config.pop('email', DEFAULT_EMAIL)
      self.email_prefs = config.pop('email_prefs', DEFAULT_EMAIL_PREFS)
      self.username = config.pop('username')
      self.cpu_total_slots = config.pop('cpu_total_slots')
      self.cpu_slots_per_job = config.pop('cpu_slots_per_job')
      self.gpu_total_slots = config.pop('gpu_total_slots')
      self.gpu_slots_per_job = config.pop('gpu_slots_per_job')
      self.mem_ram = config.pop('mem_ram')
    except:
      print('Missing a meta-configuration parameter')
      sys.exit(1)


def dict_append(d1, d2):
  assert not set(d1.keys()).intersection(set(d2.keys())), \
    "can't append overlapping dictionaries: d1={}, d2={}".format(d1, d2)

  out = d1.copy()
  out.update(d2)

  return out


def enumerate_param_combs(tree):
  var_params = []  # parameters whose values will be swept over
  fixed_params = []  # parameters with only one value
  to_recurse = []

  for key, value in tree.items():
    assert type(key) in [str], \
      "Found non-str variable key: %s of type %s" % (key, type(key))

    if key in list_arguments:
      if type(value[0]) == list:
        var_params.append((key, value))
      else:
        fixed_params.append((key, value))
    else:
      if type(value) == list:
        is_dict = [type(subval) == dict for subval in value]
        if all(is_dict):
          # this parameter has subparameters
          to_recurse.append(value)
        elif any(is_dict):
          raise ValueError(
            "can't have mixed dict and non-dict values")
        else:
          var_params.append((key, value))
      # elif type(value) in [str, float, int, bool] or value is None:
      # modified for new encoders.json
      elif type(value) in [str, float, int, bool, dict] or value is None:
        fixed_params.append((key, value))
      else:
        raise ValueError("Found non-list, non-str variable: \
                                %s : %s(in type %s) " % (key, value,
                                                         type(value)))

  recursed_args = []
  for value in to_recurse:
    subtree_args = []
    for subtree in value:
      subtree_args.extend(enumerate_param_combs(subtree))
    recursed_args.append(subtree_args)

  recursed_product = list(itertools.product(*recursed_args))
  if recursed_product:
    recursed_product = [
      reduce((lambda x, y: dict_append(x, y)), lst, {})
      for lst in recursed_product
    ]

  fixed_dict = dict(fixed_params)
  var_param_keys = [key for key, _ in var_params]
  var_param_vals = [val for _, val in var_params]
  var_param_values_product = list(itertools.product(*var_param_vals))

  args = []
  for var_param_values_comb in var_param_values_product:
    var_param_comb = dict(zip(var_param_keys, var_param_values_comb))
    for recursed_val in recursed_product:
      arg_dict = dict_append(
        dict_append(fixed_dict, var_param_comb), recursed_val)
      args.append(arg_dict)

  return args


def name_from_params(params, mode=None):
  """In this script there's no grid search and experiment name consists of
dataset name, args path and architecture and be easily found and used
later on(in test/predict/finetune modes)
"""

  # ''' Hash a dict of params into a ~unique id '''

  # assert params.get('name') is None, "Can't name already-named params"

  # bytes_params = bytes(json.dumps(sorted(params.items())), 'ascii')
  # int_hash = int(hashlib.md5(bytes_params).hexdigest(), 16)

  # assert all((params.get(key) is not None for key in keys)), \
  #     "params was missing one of {}".format(keys)

  # key_name = '-'.join((params[key] for key in keys))
  # name = '{}--{:d}'.format(key_name, int_hash)

  """
  mode:
      None: common name
      init: pre-trained model's name
      fine-tune: fine-tuned model's name
  """
  if mode == 'init':
    name_list = ['datasets', 'init_dataset_suffix',
                 'args_paths', 'architecture']
  else:
    name_list = ['dataset_name',
                 'args_paths', 'architecture']
  name = '/'.join([
    params[i] if 'suffix' not in i else params[i][1:]
    for i in name_list])

  return name


def flags_from_params(params, key_prefix='\\\n    --', key_suffix=''):
  pairs = [
    '{}{} {} {}'.format(key_prefix, key, value, key_suffix)
    for key, value in params.items()
  ]

  return ' '.join(pairs)


def write_exp_bash_script(temp_script_filename, meta_config, exp_params_comb):
  shell_command = ''

  exp_flags = flags_from_params(exp_params_comb)

  os.makedirs(os.path.dirname(temp_script_filename), exist_ok=True)
  with open(temp_script_filename, 'w') as f:
    if meta_config.cpu_or_gpu == 'gpu':
      # Loads the necessary environment variables from .bashrc
      shell_command += 'source ' + meta_config.bashrc_path + '\n'
      shell_command += '{}\n'.format(meta_config.gpu_venv)
    else:
      shell_command += '{}\n'.format(meta_config.cpu_venv)
    shell_command += 'cd {}\n'.format(meta_config.root_dir)
    if meta_config.cpu_or_gpu == 'gpu':
      shell_command += 'CUDA_VISIBLE_DEVICES=`free-gpu` '
    shell_command += 'python {} {}'.format(meta_config.code_path,
                                           exp_flags)
    f.write(shell_command)

    # print('script filename: ' + temp_script_filename)
    # print('script:')
    print(shell_command)
    print('\n\n')


def qstat(username):
  ''' Retrieve data for available slots '''
  cmd = 'qstat -u {}'.format(username)
  cmd = cmd.split()

  lines = subprocess.check_output(cmd).decode('utf8').rstrip().split('\n')

  slot_usage = dict()
  if len(lines) > 1:
    for line in lines:
      if username in line:
        job_feats = line.split()
        jobnum = int(job_feats[JOBNUM_IDX])
        slots = int(job_feats[NUMSLOTS_IDX])
        slot_usage[jobnum] = slots

  return slot_usage


def qsub(qsub_params, temp_script_file):
  qsub_flags = flags_from_params(qsub_params, key_prefix='-')
  cmd = 'qsub {} {}'.format(qsub_flags, temp_script_file)

  print(cmd)
  output = subprocess.check_output(cmd.split()).decode('utf8')
  return output


def create_qsub_params(meta_config):
  qsub_params = {
    # 'e': os.path.join(meta_config.results_dir, 'e'),
    # 'o': os.path.join(meta_config.results_dir, 'o'),
    'l':
      'mem_free={:d}G,ram_free={:d}G'.format(meta_config.mem_ram,
                                             meta_config.mem_ram)
  }

  if meta_config.cpu_or_gpu == 'cpu':
    slots_per_job = meta_config.cpu_slots_per_job
  else:
    slots_per_job = meta_config.gpu_slots_per_job

  if slots_per_job > 1:
    qsub_params['pe smp'] = slots_per_job

  if meta_config.cpu_or_gpu == 'gpu':
    # NOTE hostnames b11-18 have the tesla K80's; could allow others
    #   c* has GTX 1080's
    # See: http://wiki.clsp.jhu.edu/view/GPUs_on_the_grid
    # qsub_params['l'] += ',gpu=1,hostname=b1[123456789]*'
    qsub_params['l'] += ',gpu=1,hostname=b1[12345678]*|c*'
    # qsub_params['l'] += ',gpu=1,hostname=c*'
    # qsub_params['l'] += ',gpu=1'
    # qsub_params['q'] = "g.q,all.q"
    qsub_params['q'] = "g.q,all.q"
  else:
    pass
    # qsub_params['l'] += 'hostname=b*|c*'

  if meta_config.email != DEFAULT_EMAIL:
    qsub_params['M'] = meta_config.email
    qsub_params['m'] = meta_config.email_prefs

  return qsub_params


def complete_path_name(path_name_template, exp_params_comb):
  # replace = re.compile(r"\.*^[\>\<].*\>")
  replace = re.compile(r"\<([a-zA-Z\_]*)\>")
  path_parts = path_name_template.split('/')
  new_path_parts = []
  for part in path_parts:
    try:
      s = re.search(replace, part)
      field = s.group(0)  # includes the angle brackets (to be removed)
      key = s.group(1)
      try:
        part = part.replace(field, exp_params_comb[key])
        new_path_parts.append(part)
      except:
        raise ValueError("Unrecognized field: %s" % s)
    except:
      # fixed path component
      new_path_parts.append(part)
  new_path_name = os.path.join(*new_path_parts)

  # Replace the leading '/' if there was one
  if path_name_template.startswith('/') and \
    not new_path_name.startswith('/'):
    new_path_name = '/' + new_path_name

  return new_path_name


def write_encoder_file(exp_params_comb):
  """No grid search for encoders.json"""
  file_name = exp_params_comb['encoder_config_file']
  architecture_name = exp_params_comb['architecture']
  datasets = exp_params_comb['datasets'].split()

  # get each dataset's encoder configuration
  encoder_configs = dict()
  for domain in datasets:
    # tmp = exp_params_comb[dataset].copy()
    # embed_kwargs_names = tmp.pop('embed_kwargs_names').split()
    # tmp['embed_kwargs'] = {k: tmp.pop(k) for k in embed_kwargs_names}
    # extract_kwargs_names = tmp.pop('extract_kwargs_names').split()
    # tmp['extract_kwargs'] = {k: tmp.pop(k) for k in extract_kwargs_names}
    # encoder_configs[dataset] = tmp.copy()
    tmp = exp_params_comb[domain].copy()
    tmp['embed_kwargs'] = tmp.pop('embed_kwargs')
    tmp['extract_kwargs'] = tmp.pop('extract_kwargs')
    encoder_configs[domain] = tmp.copy()

    # construct the overall configuration
  configuration = dict()
  configuration[architecture_name] = dict()
  configuration[architecture_name]['embedders_tied'] = \
    exp_params_comb['embedders_tied']
  configuration[architecture_name]['extractors_tied'] = \
    exp_params_comb['extractors_tied']

  for domain in datasets:
    configuration[architecture_name][domain] = encoder_configs[domain]

  if configuration[architecture_name]['embedders_tied']:
    # assert that embedder configs are the same
    assert all([
      configuration[architecture_name][a]['embed_fn'] ==
      configuration[architecture_name][b]['embed_fn'] for a in datasets
      for b in datasets
    ])
    assert all([
      configuration[architecture_name][a]['embed_kwargs'] ==
      configuration[architecture_name][b]['embed_kwargs']
      for a in datasets for b in datasets
    ])

  if configuration[architecture_name]['extractors_tied']:
    # assert that extractor configs are the same
    assert all([
      configuration[architecture_name][a]['extract_fn'] ==
      configuration[architecture_name][b]['extract_fn'] for a in datasets
      for b in datasets
    ])
    assert all([
      configuration[architecture_name][a]['extract_kwargs'] ==
      configuration[architecture_name][b]['extract_kwargs']
      for a in datasets for b in datasets
    ])

  os.makedirs(os.path.dirname(file_name), exist_ok=True)
  with open(file_name, 'w') as f:
    json.dump(configuration, f)


def run_single_experiment(meta_config, exp_params_comb, qsub_params, debug):
  # Generate the name for this job, based on params
  # datset_name first as others depend on it
  if exp_params_comb['mode'] == 'finetune':
    exp_params_comb['finetune_dataset_name'] = complete_path_name(
      exp_params_comb['finetune_dataset_name'], exp_params_comb)
    exp_params_comb['finetune_dataset_name'] = complete_path_name(
      exp_params_comb['finetune_dataset_name'], exp_params_comb)
    exp_params_comb['finetune_dataset_name'] = complete_path_name(
      exp_params_comb['finetune_dataset_name'], exp_params_comb)
    exp_params_comb['dataset_name'] = exp_params_comb['finetune_dataset_name']
  else:
    exp_params_comb['dataset_name'] = complete_path_name(
      exp_params_comb['dataset_name'], exp_params_comb)
    exp_params_comb['dataset_name'] = complete_path_name(
      exp_params_comb['dataset_name'], exp_params_comb)

  if exp_params_comb['mode'] == 'finetune':
    init_name = name_from_params(exp_params_comb, mode='init')
    exp_params_comb['init_name'] = init_name
  name = name_from_params(exp_params_comb)
  exp_params_comb['name'] = name

  # Fill out any arguments that depend on other fields' values
  # res_dir = complete_path_name(meta_config.results_dir, exp_params_comb)

  results_dir = complete_path_name(meta_config.results_dir, exp_params_comb)
  results_dir = results_dir.replace('<root_dir>', meta_config.root_dir)

  qsub_params = {
    'e':
      os.path.join(results_dir, 'e'),
    'o':
      os.path.join(results_dir, 'o'),
    'l':
      'mem_free={:d}G,ram_free={:d}G'.format(meta_config.mem_ram,
                                             meta_config.mem_ram)
  }



  for field in exp_params_comb:
    if isinstance(exp_params_comb[field], str):
      exp_params_comb[field] = exp_params_comb[field].replace(
        '<root_dir>', meta_config.root_dir)
      exp_params_comb[field] = complete_path_name(
        exp_params_comb[field], exp_params_comb)

  if os.path.exists(exp_params_comb['log_file']):
    # job has already run/started
    print('DONE: {} exists'.format(exp_params_comb['log_file']))
    return False
  else:
    pass

  write_encoder_file(exp_params_comb)

  # Write bash file to execute experiment
  exp_params_comb = remove_extra_fields(exp_params_comb)
  temp_script_file = os.path.join(results_dir,
                                  '{}.sh'.format(exp_params_comb['mode']))
  write_exp_bash_script(temp_script_file, meta_config, exp_params_comb)

  # Submit the job if not in debug mode
  if debug:
    return

  output = qsub(qsub_params, temp_script_file)
  print('DOING: {}'.format(output.rstrip()))

  success = re.compile(r"Your job (\d+) \([^\)]*\) has been submitted")
  if success.search(output):
    return True
  else:
    return False


def remove_extra_fields(exp_params_comb):
  # Remove dataset-specific encoder information before writing bash script
  datasets = exp_params_comb['datasets'].split()
  for domain in datasets:
    exp_params_comb.pop(domain)

  # Remove extraneous information
  fields_to_remove = [
    'expt_setup_name', 'embedders_tied', 'extractors_tied', 'dataset_name',
    'name', 'datasets', 'dataset_suffixes', 'args_paths', 'subdirs',
    'eval_dirs', 'text_types', 'lang', 'subdir', 'init_dataset_suffix',
    'init_dataset_name', 'finetune_dataset_suffix', 'init_name',
    'finetune_dataset_name'
  ]

  # mode train etc.
  for field in fields_to_remove:
    if field in exp_params_comb:
      exp_params_comb.pop(field)

  return exp_params_comb


def consistent(encoders, exp_comb):
  # Determine whether encoders' hyperparameters are consistent with each other

  if len(encoders) == 1:
    # an encoder is automatically consistent with itself
    return True

  embedders_tied = exp_comb['embedders_tied']
  extractors_tied = exp_comb['extractors_tied']

  embed_fns = set([enc['embed_fn'] for enc in encoders])
  extract_fns = set([enc['extract_fn'] for enc in encoders])

  if embedders_tied:
    if len(embed_fns) != 1:
      # different embedding functions
      return False

    embed_kwargs_names = set(
      [enc['embed_kwargs_names'] for enc in encoders])
    if len(embed_kwargs_names) != 1:
      # different kinds of embedder arguments
      return False

    all_embed_kwargs = list()
    for enc in encoders:
      embed_kwargs_names = enc['embed_kwargs_names'].split()
      embed_kwargs = {e_k_n: enc[e_k_n] for e_k_n in embed_kwargs_names}
      all_embed_kwargs.append(embed_kwargs)

    if not all(
      [d1 == d2 for d1 in all_embed_kwargs for d2 in all_embed_kwargs]):
      # different embedder argument values
      return False

  if extractors_tied:
    if len(extract_fns) != 1:
      # different extractor functions
      return False

    extract_kwargs_names = set(
      [enc['extract_kwargs_names'] for enc in encoders])
    if len(extract_kwargs_names) != 1:
      # different kinds of extractor arguments
      return False

    all_extract_kwargs = list()
    for enc in encoders:
      extract_kwargs_names = enc['extract_kwargs_names'].split()
      extract_kwargs = {
        e_k_n: enc[e_k_n]
        for e_k_n in extract_kwargs_names
      }
      all_extract_kwargs.append(extract_kwargs)

    if not all([
      d1 == d2 for d1 in all_extract_kwargs
      for d2 in all_extract_kwargs
    ]):
      # different extractor argument values
      return False

  return True


def dict2hash(d):
  # Convert a dictionary into a hash
  # This should be unique up to key names
  contents = []
  for k, v in d.items():
    if isinstance(v, collections.Hashable):
      contents.append(v)
    elif type(v) == dict:
      contents.append(tuple([dict2hash(v)]))
  return hash(tuple(contents))


def merge_combinations(exp_params_combs_list, encoder_params_combs_list):
  # Combine experimental hyperparameters with an encoder per dataset

  # remove disallowed combinations
  # e.g., those that have `share_embedder`=True
  #       but have different settings for their embedder

  # all allowable combinations of experimental hyperparameters
  # and encoder hyperparameters
  prod = list()
  hashes = set()

  for exp_comb in tqdm(exp_params_combs_list):
    datasets = exp_comb['datasets'].split()

    # Each element of this list is itself a list of (dataset, encoder) pairs
    # At this point, all hyperparameter values have been selected
    dataset_encoder_pairs_list = list(
      itertools.product(*[
        itertools.product((d,), encoder_params_combs_list)
        for d in datasets
      ]))
    for pairs in dataset_encoder_pairs_list:
      encoders = [p[1] for p in pairs]
      if not consistent(encoders, exp_comb):
        # inconsistent
        continue
      else:
        # Combine hyperparameters with encoder hyperparameters
        # into a single data structure
        new_comb = dict()
        new_comb = exp_comb.copy()
        for pair in pairs:
          d = pair[0]
          e = pair[1]
          new_comb[d] = e
          hash_ = dict2hash(new_comb)
        if hash_ in hashes:
          # We've seen this combination before. Don't add it again
          break
        else:
          # this is a new combination
          prod.append(new_comb)

  return prod


def run_all_experiments(meta_config, exp_config, encoder_config):
  debug = meta_config.debug
  exp_params_combs_list = enumerate_param_combs(exp_config)
  encoder_params_combs_list = enumerate_param_combs(encoder_config)

  exp_params_combs_list = merge_combinations(exp_params_combs_list,
                                             encoder_params_combs_list)

  for field in ['results_dir']:
    # complete meta_config fields that depend on the values of other fields
    # NOTE: this assumes that `field` has the same value for all elements
    #       of exp_params_combs_list
    setattr(
      meta_config, field,
      complete_path_name(
        getattr(meta_config, field), exp_params_combs_list[0]))

  qsub_params = create_qsub_params(meta_config)

  slots_per_job = {
    'cpu': meta_config.cpu_slots_per_job,
    'gpu': meta_config.gpu_slots_per_job,
  }[meta_config.cpu_or_gpu]
  total_slots = {
    'cpu': meta_config.cpu_total_slots,
    'gpu': meta_config.gpu_total_slots,
  }[meta_config.cpu_or_gpu]

  if slots_per_job > 1:
    qsub_params['pe smp'] = slots_per_job

  if meta_config.debug:
    for exp_params_comb in exp_params_combs_list:
      run_single_experiment(
        meta_config, exp_params_comb, qsub_params, debug=debug)
    return

  # slot_usage = qstat(meta_config.username)
  # available_slots = total_slots - sum(slot_usage.values())
  # jobs_to_run = available_slots // slots_per_job

  available_slots = total_slots
  jobs_to_run = available_slots // slots_per_job

  jobs_run = 0
  print('STARTING: running up to {} of {} total jobs'.format(
    jobs_to_run, len(exp_params_combs_list)))

  for exp_params_comb in exp_params_combs_list:
    success = run_single_experiment(
      meta_config, exp_params_comb, qsub_params, debug=debug)

    if success:
      jobs_run += 1
      if meta_config.cpu_or_gpu == 'gpu':
        # avoid gpu race conditions
        time.sleep(10)

    if jobs_run >= jobs_to_run:
      break

  print('EXITING: ran {} jobs'.format(jobs_run))


def main():
  if len(sys.argv) < NUM_ARGS + 1:
    print('Usage: python qsub_stl_jobs.py '
          '<hyperparameter-config-file> <encoder-config-file>')
    sys.exit(1)

  config_file = sys.argv[1]
  config = None
  with open(config_file) as f:
    # allow for '//' comments
    lines = [line for line in f if not line.lstrip().startswith('/')]
    # print(' '.join(lines))
    config = json.loads(' '.join(lines))
  encoder_config_file = sys.argv[2]
  encoder_config = None
  with open(encoder_config_file) as f:
    # allow for '//' comments
    lines = [line for line in f if not line.lstrip().startswith('/')]
    encoder_config = json.loads(' '.join(lines))

  # print(encoder_config)

  config = load_json(sys.argv[1])

  # remove unnecesary arguments

  finetune_specific_fields = [
    'checkpoint_dir_init'
  ]


  if config['mode'] != 'finetune':
    for i in finetune_specific_fields:
      if i in config:
        config.pop(i)

  # No comments in encod config file generated by get_qsub_encod.py
  encoder_config = json.load(open(sys.argv[2]))[config['architecture']]

  meta_config = MetaConfig(config)
  # config now contains only experimental hyperparameters
  # (i.e., no "meta" parameters [venv, slots_per_job, etc.])
  run_all_experiments(meta_config, config, encoder_config)


if __name__ == '__main__':
  main()

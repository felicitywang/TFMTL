import collections
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys

from functools import reduce
from tqdm import tqdm

NUM_ARGS = 2
JOBNUM_IDX = 0
NUMSLOTS_IDX = -1

DEFAULT_EMAIL = ''
DEFAULT_EMAIL_PREFS = 'n'

'''
        venv: command to activate (anaconda) virtual environment
        root: root directory of code
        module: module to be executed from root by 'python -m'
        jobs_dir: where to store .sh files and qsub output
        results_dir: where to save output json
        email: address to send email (optional)
        email_prefs: preferences for email notifications (optional)
        username: grid username for qstat
        total_slots: total number of slots to dedicate to experiments
                     (must not exceed your quota)
        slots_per_job: number of slots to dedicate to each experiment instance
        mem_ram: memory to use per job (in GB)

        exp_params: params for the module run
'''


class MetaConfig(object):
    def __init__(self, config):
        try:
            self.venv = config.pop('venv')
            self.root = config.pop('root')
            self.module = config.pop('module')
            self.jobs_dir = config.pop('jobs_dir')
            self.results_dir = config.pop('results_dir')
            self.email = config.pop('email', DEFAULT_EMAIL)
            self.email_prefs = config.pop('email_prefs', DEFAULT_EMAIL_PREFS)
            self.username = config.pop('username')
            self.total_slots = config.pop('total_slots')
            self.slots_per_job = config.pop('slots_per_job')
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

        if type(value) == list:
            is_dict = [type(subval) == dict for subval in value]
            if all(is_dict):
                # this parameter has subparameters
                to_recurse.append(value)
            elif any(is_dict):
                raise ValueError("can't have mixed dict and non-dict values")
            else:
                var_params.append((key, value))
        elif type(value) in [str, float, int, bool] or value is None:
            fixed_params.append((key, value))
        else:
            raise ValueError("Found non-list, non-str variable: \
                              %s : %s" % (key, value))

    recursed_args = []
    for value in to_recurse:
        subtree_args = []
        for subtree in value:
            subtree_args.extend(enumerate_param_combs(subtree))
        recursed_args.append(subtree_args)

    recursed_product = list(itertools.product(*recursed_args))
    if recursed_product:
        recursed_product = [reduce((lambda x, y: dict_append(x, y)), lst, {})
                            for lst in recursed_product]

    fixed_dict = dict(fixed_params)
    var_param_keys = [key for key, _ in var_params]
    var_param_vals = [val for _, val in var_params]
    var_param_values_product = list(itertools.product(*var_param_vals))

    args = []
    for var_param_values_comb in var_param_values_product:
        var_param_comb = dict(zip(var_param_keys, var_param_values_comb))
        for recursed_val in recursed_product:
            arg_dict = dict_append(dict_append(fixed_dict,
                                               var_param_comb),
                                   recursed_val)
            args.append(arg_dict)

    return args


def name_from_params(params, keys=['model']):
    ''' Hash a dict of params into a ~unique id '''

    assert params.get('name') is None, "Can't name already-named params"

    bytes_params = bytes(json.dumps(sorted(params.items())), 'ascii')
    int_hash = int(hashlib.md5(bytes_params).hexdigest(), 16)

    assert all((params.get(key) is not None for key in keys)), \
        "params was missing one of {}".format(keys)

    key_name = '-'.join((params[key] for key in keys))
    name = '{}--{:d}'.format(key_name, int_hash)

    return name


def flags_from_params(params, key_prefix='--'):
    pairs = ['{}{} {}'.format(key_prefix, key, value)
             for key, value in params.items()]
    return ' '.join(pairs)


def write_exp_bash_script(temp_script_filename, meta_config, exp_params_comb):
    exp_flags = flags_from_params(exp_params_comb)
    os.makedirs(os.path.dirname(temp_script_filename), exist_ok=True)
    with open(temp_script_filename, 'w') as f:
        f.write('{}\n'.format(meta_config.venv))
        f.write('cd {}\n'.format(meta_config.root))
        f.write('python -m {} {}'.format(meta_config.module, exp_flags))


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
        'o': meta_config.jobs_dir,
        'e': meta_config.jobs_dir,
        'pe smp': meta_config.slots_per_job,
        'l': 'mem_free={:d}G,ram_free={:d}G'.format(meta_config.mem_ram,
                                                    meta_config.mem_ram)
      }

    if meta_config.email != DEFAULT_EMAIL:
        qsub_params['M'] = meta_config.email
        qsub_params['m'] = meta_config.email_prefs

    return qsub_params


def complete_path_name(path_name_template, exp_params_comb):
    replace = re.compile(r"\<(.*)\>")
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
  file_name = exp_params_comb['encoder_config_file']
  architecture_name = exp_params_comb['architecture']
  datasets = exp_params_comb['datasets'].split()

  # get each dataset's encoder configuration
  encoder_configs = dict()
  for dataset in datasets:
    tmp = exp_params_comb[dataset].copy()
    embed_kwargs_names = tmp.pop('embed_kwargs_names').split()
    tmp['embed_kwargs'] = {k: tmp.pop(k) for k in embed_kwargs_names}
    extract_kwargs_names = tmp.pop('extract_kwargs_names').split()
    tmp['extract_kwargs'] = {k: tmp.pop(k) for k in extract_kwargs_names}
    encoder_configs[dataset] = tmp.copy()

  # construct the overall configuration
  configuration = dict()
  configuration[architecture_name] = dict()
  configuration[architecture_name]['embedders_tied'] = \
      exp_params_comb['embedders_tied']
  configuration[architecture_name]['extractors_tied'] = \
      exp_params_comb['extractors_tied']

  for dataset in datasets:
    configuration[architecture_name][dataset] = encoder_configs[dataset]

  if configuration[architecture_name]['embedders_tied']:
    # assert that embedder configs are the same
    assert all([configuration[architecture_name][a]['embed_fn'] ==
                configuration[architecture_name][b]['embed_fn']
                for a in datasets
                for b in datasets])
    assert all([configuration[architecture_name][a]['embed_kwargs'] ==
                configuration[architecture_name][b]['embed_kwargs']
                for a in datasets
                for b in datasets])

  if configuration[architecture_name]['extractors_tied']:
    # assert that extractor configs are the same
    assert all([configuration[architecture_name][a]['extract_fn'] ==
                configuration[architecture_name][b]['extract_fn']
                for a in datasets
                for b in datasets])
    assert all([configuration[architecture_name][a]['extract_kwargs'] ==
                configuration[architecture_name][b]['extract_kwargs']
                for a in datasets
                for b in datasets])

  os.makedirs(os.path.dirname(file_name), exist_ok=True)
  with open(file_name, 'w') as f:
    json.dump(configuration, f)


def run_single_experiment(meta_config,
                          exp_params_comb,
                          qsub_params,
                          debug=False):
    # Generate the name for this job, based on params
    name = name_from_params(exp_params_comb, keys=["expt_setup_name"])
    exp_params_comb['name'] = name

    # Fill out any arguments that depend on other fields' values
    # res_dir = complete_path_name(meta_config.results_dir, exp_params_comb)
    jobs_dir = complete_path_name(meta_config.jobs_dir, exp_params_comb)
    for field in ['checkpoint_dir', 'log_file', 'encoder_config_file']:
      exp_params_comb[field] = complete_path_name(exp_params_comb[field],
                                                  exp_params_comb)

    if os.path.exists(exp_params_comb['log_file']):
        # job has already run/started
        print('DONE: {} exists'.format(exp_params_comb['log_file']))
        return False
    else:
        pass

    if debug:
      print(exp_params_comb)
      print('---')

    write_encoder_file(exp_params_comb)

    # Remove dataset-specific encoder information before writing bash script
    datasets = exp_params_comb['datasets'].split()
    for dataset in datasets:
      exp_params_comb.pop(dataset)

    # Remove extraneous information
    for field in ['expt_setup_name',
                  'embedders_tied',
                  'extractors_tied',
                  'name']:
      exp_params_comb.pop(field)

    # Write bash file to execute experiment
    temp_script_file = os.path.join(jobs_dir, '{}.sh'.format(name))
    write_exp_bash_script(temp_script_file, meta_config, exp_params_comb)

    # Submit the job
    output = qsub(qsub_params, temp_script_file)
    print('DOING: {}'.format(output.rstrip()))

    success = re.compile(r"Your job (\d+) \([^\)]*\) has been submitted")
    if success.search(output):
        return True
    else:
        return False


def consistent(encoders, exp_comb):
  # Determine whether encoders' hyperparameters are consistent with each other

  if len(encoders) == 1:
    # an encoder is automatically consistent with itself
    return True

  embedders_tied = exp_comb['embedders_tied']
  extractors_tied = exp_comb['extractors_tied']

  embed_fns = set([enc['embed_fn'] for enc in encoders])
  extract_fns = set([enc['extract_fn'] for enc in encoders])

  # TODO(seth): refactor this check to avoid code duplication
  if embedders_tied:
    if len(embed_fns) != 1:
      # different embedding functions
      return False

    embed_kwargs_names = set([enc['embed_kwargs_names'] for enc in encoders])
    if len(embed_kwargs_names) != 1:
      # different kinds of embedder arguments
      return False

    all_embed_kwargs = list()
    for enc in encoders:
      embed_kwargs_names = enc['embed_kwargs_names'].split()
      embed_kwargs = {e_k_n: enc[e_k_n] for e_k_n in embed_kwargs_names}
      all_embed_kwargs.append(embed_kwargs)

    if not all([d1 == d2
                for d1 in all_embed_kwargs
                for d2 in all_embed_kwargs]):
      # different embedder argument values
      return False

  if extractors_tied:
    if len(extract_fns) != 1:
      # different extractor functions
      return False

    extract_kwargs_names = set([enc['extract_kwargs_names']
                                for enc in encoders])
    if len(extract_kwargs_names) != 1:
      # different kinds of extractor arguments
      return False

    all_extract_kwargs = list()
    for enc in encoders:
      extract_kwargs_names = enc['extract_kwargs_names'].split()
      extract_kwargs = {e_k_n: enc[e_k_n] for e_k_n in extract_kwargs_names}
      all_extract_kwargs.append(extract_kwargs)

    if not all([d1 == d2
                for d1 in all_extract_kwargs
                for d2 in all_extract_kwargs]):
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
    dataset_encoder_pairs_list = list(itertools.
                                      product(*[itertools.
                                                product(
                                                  (d,),
                                                  encoder_params_combs_list
                                                )
                                                for d in datasets]))
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


def run_all_experiments(meta_config, exp_config, encoder_config, debug=False):
    exp_params_combs_list = enumerate_param_combs(exp_config)
    encoder_params_combs_list = enumerate_param_combs(encoder_config)

    exp_params_combs_list = merge_combinations(exp_params_combs_list,
                                               encoder_params_combs_list)

    for field in ['jobs_dir']:
      # complete meta_config fields that depend on the values of other fields
      # NOTE: this assumes that `field` has the same value for all elements
      #       of exp_params_combs_list
      setattr(meta_config,
              field,
              complete_path_name(getattr(meta_config, field),
                                 exp_params_combs_list[0]))

    qsub_params = create_qsub_params(meta_config)

    slot_usage = qstat(meta_config.username)
    available_slots = meta_config.total_slots - sum(slot_usage.values())
    jobs_to_run = available_slots // meta_config.slots_per_job

    jobs_run = 0
    print('STARTING: running up to {} of {} total jobs'.
          format(jobs_to_run,
                 len(exp_params_combs_list)))
    for exp_params_comb in exp_params_combs_list:
        success = run_single_experiment(meta_config,
                                        exp_params_comb,
                                        qsub_params,
                                        debug=debug)

        if success:
            jobs_run += 1

        if jobs_run >= jobs_to_run:
            break

    print('EXITING: ran {} jobs'.format(jobs_run))


def main():
    if len(sys.argv) < NUM_ARGS + 1:
        print('Usage: python gridsearch_mtl.py '
              '<hyperparameter-config-file> <encoder-config-file>')
        sys.exit(1)

    config_file = sys.argv[1]
    config = None
    with open(config_file) as f:
        config = json.loads(' '.join(f.readlines()))

    encoder_config_file = sys.argv[2]
    encoder_config = None
    with open(encoder_config_file) as f:
      encoder_config = json.loads(' '.join(f.readlines()))

    debug = False

    mc = MetaConfig(config)

    # config now contains only experimental hyperparameters
    # (i.e., no "meta" parameters [venv, slots_per_job, etc.])
    run_all_experiments(mc, config, encoder_config, debug=debug)


if __name__ == '__main__':
    main()

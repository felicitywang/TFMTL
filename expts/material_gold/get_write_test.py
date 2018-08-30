TASKS = [
  'GOV_100',
  'LIF_100',
  'HEA_100',
  'LAW_100',
  'BUS_100',
  'MIL_100',
  'GOV_1000',
  'LIF_1000',
  'HEA_1000',
  'LAW_1000',
  'BUS_1000',
  'MIL_1000'
]


def main():
  args_file = 'args_nopretrain.json'
  dataset_path = '/min_1_max_-1_vocab_-1_doc_1000/'
  dir = 'pilot_2_gold_1B'
  suffix = '_50.0'
  for task in TASKS:
    job = 'python ../scripts/write_tfrecords_test.py ' \
          + args_file + \
          ' data/json/' + dir + '/' + task[:3] + suffix + \
          ' data/tf/' + dir + '/' + task[:3] + suffix + \
          ' data/tf/single/' + task + dataset_path
    print(job)


if __name__ == '__main__':
  main()

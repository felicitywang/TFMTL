import os

from mtl.util.util import make_dir

DEBUG = False

TASKS = [
  # 'GOV_100',
  # 'LIF_100',
  # 'HEA_100',
  # 'LAW_100',
  # 'BUS_100',
  # 'MIL_100',
  'GOV_1000',
  'LIF_1000',
  'HEA_1000',
  'LAW_1000',
  'BUS_1000',
  'MIL_1000'
]

CLASS_SIZE = 2

METRICS = [
  'Acc',
  'F1_PosNeg_Macro',
  # 'Precision_Macro',
  # 'Recall_Macro',
  # 'Confusion_Matrix'
]

SYNTHETIC = True


def make_job(dataset, prefix, seed, architecture, test_log_name):
  # dataset_paths = os.path.join('data/tf/single', dataset, prefix)
  if SYNTHETIC:
    dataset_paths = os.path.join('data/tf/pilot_1_synthetic', dataset[:3] +
                                 '_50.0', prefix)
  else:
    dataset_paths = os.path.join('data/tf/gold_sent', dataset[:3], prefix)
  alphas = '1'
  vocab_size_file = os.path.join('data/tf/single', dataset, prefix,
                                 'vocab_size.txt')
  topic_paths = os.path.join('data/json/', dataset, 'data.json.gz')

  name = dataset + '_' + prefix + '_' + architecture + '_' + 'cpu'

  result_dir = os.path.join('results', 'seed_' + str(seed), name)
  make_dir(result_dir)

  encoder_path = 'encoders.json'

  output = "python ../scripts/discriminative_driver.py \\\n" \
           "  --model mult \\\n" + \
           "  --num_train_epochs 30 \\\n" + \
           "  --optimizer rmsprop" + " \\\n" + \
           "  --lr0 0.001" + " \\\n" + \
           "  --patience 3" + " \\\n" + \
           "  --early_stopping_acc_threshold 1.0" + " \\\n" + \
           "  --shared_mlp_layers 1" + " \\\n" + \
           "  --shared_hidden_dims 100" + " \\\n" + \
           "  --private_mlp_layers 1" + " \\\n" + \
           "  --private_hidden_dims 100" + " \\\n" + \
           "  --input_keep_prob 1.0" + " \\\n" + \
           "  --output_keep_prob 0.5" + " \\\n" + \
           "  --l2_weight 0" + " \\\n" + \
           "  --tuning_metric Acc" + " \\\n" + \
           "  --mode test" + "\\\n" + \
           "  --checkpoint_dir " + os.path.join(result_dir, 'ckpt') + " \\" \
                                                                      "\n" + \
           "  --datasets " + dataset[:dataset.find('_')] + " \\\n" + \
           "  --dataset_paths " + dataset_paths + " \\\n" + \
           "  --topics_paths " + topic_paths + " \\\n" + \
           "  --class_sizes " + str(CLASS_SIZE) + " \\\n" + \
           "  --vocab_size_file " + vocab_size_file + " \\\n" + \
           "  --encoder_config_file " + encoder_path + " \\\n" + \
           "  --architecture " + architecture + " \\\n" + \
           "  --seed " + str(seed) + " \\\n" + \
           "  --alphas " + alphas + " \\\n" + \
           "  --log_file " + test_log_name + " \\\n" + \
           "  --metrics " + ' '.join(METRICS) + " \n\n"

  print(output)


def main():
  dataset_path_prefixes = [
    # 'min_1_max_-1_vocab_-1_doc_1000',
    'min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand',
    # 'min_1_max_-1_vocab_-1_doc_400_wiki-news-300d-1M_expand',
    # 'min_1_max_-1_vocab_-1_doc_400_GoogleNews-vectors-negative300-SLIM_expand'
  ]

  import sys
  seed = sys.argv[1]
  test_log_name = sys.argv[2]

  # print('Usage: python get_jobs.py seed(integer) test_log_name')

  # architecture = sys.argv[2]
  architectures = [
    # 'bilstm_expand_glove',
    # 'bigru_expand_glove',
    'paragram_expand_glove',
    # 'cnn_expand_glove',
    # 'bilstm_expand_fasttext',
    # 'bigru_expand_fasttext',
    # 'paragram_expand_fasttext',
    # 'cnn_expand_fasttext',
    # 'bilstm_expand_word2vec_slim',
    # 'bigru_expand_word2vec_slim',
    # 'paragram_expand_word2vec_slim',
    # 'cnn_expand_word2vec_slim',
    # 'paragram_nopretrain',
    # 'bilstm_nopretrain',
    # 'bigru_nopretrain',
    # 'cnn_nopretrain'
  ]

  for dataset in TASKS:
    for prefix in dataset_path_prefixes:
      for architecture in architectures:
        make_job(dataset, prefix, seed, architecture, test_log_name)


if __name__ == "__main__":
  main()

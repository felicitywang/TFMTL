import os
import subprocess

from mtl.util.util import make_dir

RUDER_TASKS = [
    # 'MultiNLI',
    # 'ABSA-L',
    # 'ABSA-R',
    # 'Topic2',
    # 'Topic5',
    # 'Target',
    'Stance',
    # 'FNC-1',
]

RUDER_AUX_TASK_DICT = {
    'Topic2': ['FNC-1', 'MultiNLI', 'Target'],
    'Topic5': ['FNC-1', 'MultiNLI', 'ABSA-L', 'Target'],
    'Target': ['FNC-1', 'MultiNLI', 'Topic5'],
    'Stance': ['FNC-1', 'MultiNLI', 'Target'],
    'ABSA-L': ['Topic5'],
    'ABSA-R': ['Topic5', 'ABSA-L', 'Target'],
    'FNC-1': ['Stance', 'MultiNLI', 'Topic5', 'ABSA-R', 'Target'],
    'MultiNLI': ['Topic5']
}

CLASS_SIZES = {
    'Topic2': 2,
    'Topic5': 5,
    'Target': 3,
    'Stance': 3,
    'ABSA-L': 3,
    'ABSA-R': 3,
    'FNC-1': 4,
    'MultiNLI': 3
}

REPORTING_METRICS = {
    'Topic2': "Recall_Macro",
    'Topic5': "MAE_Macro",
    'Target': "F1_Macro",
    'Stance': "F1_PosNeg_Macro",
    'ABSA-L': "Acc",
    'ABSA-R': "Acc",
    'FNC-1': "Acc",
    'MultiNLI': "Acc"
}

debug = True
if debug:
    root_dir = '.'
    data_dir = 'data/tf/'
    encoder_dir = "encoder_files/"
    code_path = "../scripts/discriminative_driver.py"
    json_dir = "data/json/"

else:
    root_dir = "/export/a08/fwang/tfmtl/expts/all_EMNLP/"
    data_dir = "/export/a08/fwang/tfmtl/expts/all_EMNLP/data/tf/"
    encoder_dir = "/export/a08/fwang/tfmtl/expts/all_EMNLP/encoder_files/"
    code_path = "/export/a08/fwang/tfmtl/expts/scripts/discriminative_driver.py"
    json_dir = "/export/a08/fwang/tfmtl/expts/all_EMNLP/data/json/"

tasks = ['st', 'mt']
vocabs = ['all_0', 'train_1']


def make_job(dataset, vocab, task, init_or_expand, finetune_or_freeze):
    dataset_dir = os.path.join(data_dir, dataset + '-' + task)
    if task == 'st':
        datasets = dataset
        dataset_paths = os.path.join(dataset_dir,
                                     vocab + "_glove_" + init_or_expand, dataset)
        class_sizes = str(CLASS_SIZES[dataset])
        alphas = '1'
        topic_paths = os.path.join(json_dir, dataset, "data.json.gz")
        vocab_size_file = os.path.join(dataset_dir,
                                       vocab + "_glove_" + init_or_expand, dataset,
                                       'vocab_size.txt')
    else:
        datasets = ' '.join([dataset] + RUDER_AUX_TASK_DICT[dataset])
        dataset_paths = ' '.join([os.path.join(dataset_dir,
                                               vocab + "_glove_" + init_or_expand,
                                               aux_dataset) for aux_dataset in
                                  [dataset] + RUDER_AUX_TASK_DICT[dataset]])
        class_sizes = ' '.join([str(CLASS_SIZES[aux_dataset]) for
                                aux_dataset in
                                [dataset] + RUDER_AUX_TASK_DICT[dataset]])
        topic_paths = ' '.join(
            [os.path.join(json_dir, aux_dataset, "data.json.gz") for
             aux_dataset in
             [dataset] + RUDER_AUX_TASK_DICT[dataset]])
        vocab_size_file = os.path.join(dataset_dir,
                                       vocab + "_glove_" + init_or_expand,
                                       'vocab_size.txt')
        alpha_num = len(RUDER_AUX_TASK_DICT[dataset]) + 1
        alphas = ' '.join([str(1.0 / alpha_num)] * alpha_num)

    name = dataset + '_' + task + '_' + vocab + \
        '_' + init_or_expand + '_' + finetune_or_freeze

    result_dir = os.path.join(root_dir, 'early_results_11', dataset, name)
    make_dir(result_dir)

    job_dir = os.path.join(root_dir, 'early_jobs_11', dataset)
    make_dir(job_dir)
    job_name = 'train_' + name + '.sh'
    job_path = os.path.join(job_dir, job_name)

    encoder_path = os.path.join(encoder_dir,
                                "encoder-" + task + "-" + vocab + ".json")

    output = "#!/bin/sh\n\
#$ -cwd\n\
#$ -o " + os.path.join(result_dir, 'o') + "\n\
#$ -e " + os.path.join(result_dir, 'e') + "\n\
#$ -M cnfxwang@gmail.com\n\
#$ -l \'gpu=1,mem_free=3G,ram_free=3G,hostname=\"b1*|c*\"\''\n\
#$ -pe smp 2\n\
#$ -V\n\
#$ -q g.q\n\
\n\
\n\
  source /home/fwang/.bashrc \n\ \
  source /home/fwang/gpu/bin/activate \n\ \
\n\
  CUDA_VISIBLE_DEVICES=`free-gpu` /home/fwang/gpu/bin/python3 " + \
             code_path + ' \\\n' \
                         "  --model mult \\\n" + \
             "  --mode train \\\n" + \
             "  --num_train_epochs 30 \\\n" + \
             "  --optimizer rmsprop" + " \\\n" + \
             "  --lr0 0.001" + " \\\n" + \
             "  --patience 3" + " \\\n" + \
             "  --early_stopping_acc_threshold 0.0" + " \\\n" + \
             "  --experiment_name RUDER_NAACL_18 \\\n" + \
             "  --shared_mlp_layers 0" + " \\\n" + \
             "  --shared_hidden_dims 0" + " \\\n" + \
             "  --private_mlp_layers 1" + " \\\n" + \
             "  --private_hidden_dims 100" + " \\\n" + \
             "  --input_keep_prob 1.0" + " \\\n" + \
             "  --output_keep_prob 1.0" + " \\\n" + \
             "  --l2_weight 0" + " \\\n" + \
             "  --tuning_metric Acc" + " \\\n" + \
             "  --checkpoint_dir " + os.path.join(result_dir, 'ckpt') + " \\" \
                                                                        "\n" + \
             "  --datasets " + datasets + " \\\n" + \
             "  --dataset_paths " + dataset_paths + " \\\n" + \
             "  --topics_paths " + topic_paths + " \\\n" + \
             "  --class_sizes " + class_sizes + " \\\n" + \
             "  --vocab_size_file " + vocab_size_file + " \\\n" + \
             "  --encoder_config_file " + encoder_path + " \\\n" + \
             "  --architecture " + 'serial_birnn_stock_glove_' \
             + init_or_expand + "_" + finetune_or_freeze + " \\\n" + \
             "  --seed 11" + " \\\n" + \
             "  --alphas " + alphas + " \\\n" + \
             "  --log_file " + os.path.join(result_dir, 'log') + " \\\n" + \
             "  --reporting_metric " + REPORTING_METRICS[dataset] + " \\\n"

    print(output)

    with open(job_path, 'w') as file:
        file.write(output)

    if not debug:
        subprocess.call(["qsub", job_path])


def main():
    for task in ['mt']:
        for dataset in RUDER_TASKS:
            # for vocab in ['train_1', 'all_0']:
            for vocab in ['all_0']:
                # for init_for_expand in ['init', 'expand']:
                for init_or_expand in ['init']:
                    for finetune_or_freeze in ['finetune']:
                        make_job(dataset, vocab, task, init_or_expand,
                                 finetune_or_freeze)


if __name__ == "__main__":
    main()

import os
import subprocess

from material_constants import PRED_DIRS as dirs

from mtl.util.util import make_dir

# TODO merge with get_jobs.py

DEBUG = True

# TODO
TASKS = [
    'TURK_GOV_90_50_ORACLE',
    'TURK_LIF_90_50_ORACLE',
    'TURK_BUS_90_50_ORACLE',
    'TURK_LAW_90_50_ORACLE',
    'TURK_HEA_90_50_ORACLE',
    'TURK_MIL_90_50_ORACLE',
    'TURK_SPO_90_50_ORACLE',

    'TURK_GOV_80_50_ORACLE',
    'TURK_LIF_80_50_ORACLE',
    'TURK_BUS_80_50_ORACLE',
    'TURK_LAW_80_50_ORACLE',
    'TURK_HEA_80_50_ORACLE',
    'TURK_MIL_80_50_ORACLE',
    'TURK_SPO_80_50_ORACLE',

    'TURK_GOV_70_50_ORACLE',
    'TURK_LIF_70_50_ORACLE',
    'TURK_BUS_70_50_ORACLE',
    'TURK_LAW_70_50_ORACLE',
    'TURK_HEA_70_50_ORACLE',
    'TURK_MIL_70_50_ORACLE',
    'TURK_SPO_70_50_ORACLE',

    'TURK_GOV_60_50_ORACLE',
    'TURK_LIF_60_50_ORACLE',
    'TURK_BUS_60_50_ORACLE',
    'TURK_LAW_60_50_ORACLE',
    'TURK_HEA_60_50_ORACLE',
    'TURK_MIL_60_50_ORACLE',
    'TURK_SPO_60_50_ORACLE',

    'TURK_GOV_50_50_ORACLE',
    'TURK_LIF_50_50_ORACLE',
    'TURK_BUS_50_50_ORACLE',
    'TURK_LAW_50_50_ORACLE',
    'TURK_HEA_50_50_ORACLE',
    'TURK_MIL_50_50_ORACLE',
    'TURK_SPO_50_50_ORACLE',

]

CLASS_SIZE = 2


def make_job(dataset, prefix, seed, architecture, type, mode, pred_dir):
    dataset_paths = os.path.join('data/tf/single', dataset, prefix)
    alphas = '1'
    topic_paths = os.path.join('data/json/', dataset, 'data.json.gz')

    name = dataset + '_' + prefix + '_' + architecture + '_' + type

    domains = ['GOV', 'LIF', 'BUS', 'LAW', 'HEA', 'MIL', 'SPO']

    result_dir = os.path.join('results', 'seed_' + str(seed), name)
    make_dir(result_dir)
    for domain in domains:
        if domain in dataset:
            dataset_name = domain

    flag_l = {
        'cpu': "#$ -l \'mem_free=50G,ram_free=50G,hostname=\"b*|c*\"\'",
        'gpu': "#$ -l \'gpu=1,mem_free=50G,ram_free=50G,hostname=\"b1[12345678]*|c*\" \n#$ -q g.q"
    }

    python = {
        'cpu': '/home/fwang/cpu/bin/python ',
        'gpu': 'CUDA_VISIBLE_DEVICES=`free-gpu` /home/fwang/gpu/bin/python '
    }

    encoder_path = 'encoders.json'

    predict_tfrecord = os.path.join('data/tf/predict/',
                                    pred_dir,
                                    # dataset[:dataset.find('_')],
                                    dataset,
                                    prefix,
                                    'predict.tf')
    # predict_output_folder = os.path.join('predictions/', pred_dir,
    #                                      architecture + '_' + prefix)
    tmp = dataset.replace(dataset_name, '')
    predict_output_folder = os.path.join('predictions/', pred_dir,
                                         tmp + '_' + prefix + '_' + architecture)

    output = "#!/bin/sh\n\
#$ -cwd\n\
#$ -o pred.o\n\
#$ -e pred.e\n\
#$ -M cnfxwang@gmail.com\n" + \
             flag_l[type] + "\n\
#$ -pe smp 2\n\
#$ -V\n\
\n\
\n\
source /home/fwang/.bashrc\n\
cd /export/a08/fwang/tfmtl/expts/material_gold/\n" + \
             python[type] + \
             "../scripts/discriminative_driver.py \\\n" \
             "  --model mult \\\n" + \
             "  --shared_mlp_layers 0" + " \\\n" + \
             "  --shared_hidden_dims 0" + " \\\n" + \
             "  --private_mlp_layers 2" + " \\\n" + \
             "  --private_hidden_dims 128" + " \\\n" + \
             "  --input_keep_prob 1.0" + " \\\n" + \
             "  --output_keep_prob 0.5" + " \\\n" + \
             "  --l2_weight 0" + " \\\n" + \
             "  --checkpoint_dir " + os.path.join(result_dir, 'ckpt') + " \\" \
                                                                        "\n" + \
             "  --mode " + mode + "\\\n" + \
             "  --datasets " + dataset_name + " \\\n" + \
             "  --dataset_paths " + dataset_paths + " \\\n" + \
             "  --topics_paths " + topic_paths + " \\\n" + \
             "  --class_sizes " + str(CLASS_SIZE) + " \\\n" + \
             "  --encoder_config_file " + encoder_path + " \\\n" + \
             "  --architecture " + architecture + " \\\n" + \
             "  --seed " + str(seed) + " \\\n" + \
             "  --alphas " + alphas + " \\\n" + \
             "  --predict_dataset " + dataset_name + " \\\n" + \
             "  --predict_tfrecord " + predict_tfrecord + " \\\n" + \
             "  --predict_output_folder " + predict_output_folder + "\n"

    print(output)

    job_dir = os.path.join('jobs', 'seed_' + str(seed))
    make_dir(job_dir)
    job_name = mode + '_' + name + '.sh'
    job_path = os.path.join(job_dir, job_name)
    with open(job_path, 'w') as file:
        file.write(output)

    if not DEBUG:
        with open(job_path, 'w') as file:
            file.write(output)
        subprocess.call(["qsub", job_path])


def main():
    dataset_path_prefixes = [
        'min_1_max_-1_vocab_-1_doc_-1',
        # 'min_1_max_-1_vocab_-1_doc_400_glove.6B.300d_expand',
        # 'min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand',
        # 'min_1_max_-1_vocab_-1_doc_-1_glove.6B.300d_expand',
        # 'min_1_max_-1_vocab_-1_doc_1000_wiki-news-300d-1M_expand',
        # 'min_1_max_-1_vocab_-1_doc_1000_GoogleNews-vectors-negative300-SLIM_expand'
    ]

    import sys
    seed = sys.argv[1]
    type = sys.argv[2]
    mode = sys.argv[3]
    # print('Usage: python get_jobs.py seed(integer) type(cpu/gpu) mode('
    #       'train/test)')
    # architecture = sys.argv[2]
    architectures = [
        'meanmax_relu_nopretrain'
    ]

    # TODO
    LANG_DIR = {
        '1A': [
            'TURK_GOV_90_50_ORACLE',
            'TURK_LIF_90_50_ORACLE',
            'TURK_BUS_90_50_ORACLE',
            'TURK_LAW_90_50_ORACLE',
            'TURK_SPO_90_50_ORACLE',

            'TURK_GOV_80_50_ORACLE',
            'TURK_LIF_80_50_ORACLE',
            'TURK_BUS_80_50_ORACLE',
            'TURK_LAW_80_50_ORACLE',
            'TURK_SPO_80_50_ORACLE',

            'TURK_GOV_70_50_ORACLE',
            'TURK_LIF_70_50_ORACLE',
            'TURK_BUS_70_50_ORACLE',
            'TURK_LAW_70_50_ORACLE',
            'TURK_SPO_70_50_ORACLE',

            'TURK_GOV_60_50_ORACLE',
            'TURK_LIF_60_50_ORACLE',
            'TURK_BUS_60_50_ORACLE',
            'TURK_LAW_60_50_ORACLE',
            'TURK_SPO_60_50_ORACLE',

            'TURK_GOV_50_50_ORACLE',
            'TURK_LIF_50_50_ORACLE',
            'TURK_BUS_50_50_ORACLE',
            'TURK_LAW_50_50_ORACLE',
            'TURK_SPO_50_50_ORACLE',
        ],
        '1B': [
            'TURK_GOV_90_50_ORACLE',
            'TURK_LIF_90_50_ORACLE',
            'TURK_HEA_90_50_ORACLE',
            'TURK_MIL_90_50_ORACLE',
            'TURK_SPO_90_50_ORACLE',

            'TURK_GOV_80_50_ORACLE',
            'TURK_LIF_80_50_ORACLE',
            'TURK_HEA_80_50_ORACLE',
            'TURK_MIL_80_50_ORACLE',
            'TURK_SPO_80_50_ORACLE',

            'TURK_GOV_70_50_ORACLE',
            'TURK_LIF_70_50_ORACLE',
            'TURK_HEA_70_50_ORACLE',
            'TURK_MIL_70_50_ORACLE',
            'TURK_SPO_70_50_ORACLE',

            'TURK_GOV_60_50_ORACLE',
            'TURK_LIF_60_50_ORACLE',
            'TURK_HEA_60_50_ORACLE',
            'TURK_MIL_60_50_ORACLE',
            'TURK_SPO_60_50_ORACLE',

            'TURK_GOV_50_50_ORACLE',
            'TURK_LIF_50_50_ORACLE',
            'TURK_HEA_50_50_ORACLE',
            'TURK_MIL_50_50_ORACLE',
            'TURK_SPO_50_50_ORACLE', ],

    }

    for basedir in dirs:
        # TODO
        if 'DEV' in basedir:
            continue
        lang = basedir[basedir.find('/') + 1:basedir.rfind('/')]
        for subdir in dirs[basedir]:
            # TODO only one best
            if 'bop' in subdir:
                continue
            pred_dir = os.path.join(basedir, subdir)
            for dataset in LANG_DIR[lang]:
                for prefix in dataset_path_prefixes:
                    for architecture in architectures:
                        make_job(dataset, prefix, seed, architecture, type, mode,
                                 pred_dir)


if __name__ == "__main__":
    main()

# !/bin/sh
# $ -cwd
# $ -o results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/o
# $ -e results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/e
# $ -M cnfxwang@gmail.com
# $ -l 'mem_free=50G,ram_free=50G'
# $ -pe smp 2
# $ -V

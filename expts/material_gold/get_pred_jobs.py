import os
import subprocess


from material_constants import PRED_DIRS, PRED_DIRS_sent

# TODO
dirs = PRED_DIRS


from mtl.util.util import make_dir

# TODO merge with get_jobs.py

DEBUG = True

# TODO
TASKS = [
    'GOV_1000',
    'LIF_1000',
    # 'HEA_1000',
    # 'LAW_1000',
    # 'BUS_1000',
    'MIL_1000',
    # 'SPO_1000'
]

CLASS_SIZE = 2


def make_job(dataset, prefix, seed, architecture, type, mode, pred_dir):
    dataset_paths = os.path.join('data/tf/single', dataset, prefix)
    alphas = '1'
    topic_paths = os.path.join('data/json/', dataset, 'data.json.gz')

    name = dataset + '_' + prefix + '_' + architecture + '_' + type

    result_dir = os.path.join('results', 'seed_' + str(seed), name)
    make_dir(result_dir)

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
                                    dataset[:dataset.find('_')],
                                    prefix,
                                    'predict.tf')
    # predict_output_folder = os.path.join('predictions/', pred_dir,
    #                                      architecture + '_' + prefix)
    predict_output_folder = os.path.join('predictions/', pred_dir,
                                         prefix + '_' + architecture)

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
             "  --datasets " + dataset[:dataset.find('_')] + " \\\n" + \
             "  --dataset_paths " + dataset_paths + " \\\n" + \
             "  --topics_paths " + topic_paths + " \\\n" + \
             "  --class_sizes " + str(CLASS_SIZE) + " \\\n" + \
             "  --encoder_config_file " + encoder_path + " \\\n" + \
             "  --architecture " + architecture + " \\\n" + \
             "  --seed " + str(seed) + " \\\n" + \
             "  --alphas " + alphas + " \\\n" + \
             "  --predict_dataset " + dataset[:dataset.find('_')] + " \\\n" + \
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
        'min_1_max_-1_vocab_-1_doc_1000',
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
        # 'bilstm_expand_glove',
        # 'bigru_expand_glove',
        # 'maxpool_relu_expand_glove',
        # 'cnn_expand_glove',
        # 'bilstm_expand_fasttext',
        # 'bigru_expand_fasttext',
        # 'maxpool_relu_expand_fasttext',
        # 'cnn_expand_fasttext',
        # 'bilstm_expand_word2vec_slim',
        # 'bigru_expand_word2vec_slim',
        # 'maxpool_relu_expand_word2vec_slim',
        # 'cnn_expand_word2vec_slim',
        # 'maxpool_relu_nopretrain',
        # 'bilstm_nopretrain',
        # 'bigru_nopretrain',
        # 'cnn_nopretrain',
        # 'bilstm_attention_nopretrain',
        # 'bigru_attention_nopretrain',
        # 'meanpool_linear_nopretrain',
        # 'meanpool_relu_nopretrain',
        # 'meanpool_tanh_nopretrain',
        # 'meanpool_linear_relu_nopretrain',
        # 'meanpool_relu_relu_nopretrain',
        # 'maxpool_relu_nopretrain',
        # 'maxpool_tanh_nopretrain',
        # 'meanmax_relu_nopretrain',
        # 'meanmax_linear_relu_nopretrain',
        # 'meanpool_relu_glove_expand',
        # 'meanpool_relu_fasttext_expand',
        # 'meanpool_relu_word2vec_slim_expand',
        # 'meanmax_relu_glove_expand',
        # 'meanmax_relu_fasttext_expand',
        # 'meanmax_relu_word2vec_slim_expand',
        'meanmax_relu_nopretrain'
    ]

    # TODO
    # LANG_DIR = {
    #     '1A': ['GOV_1000', 'LIF_1000', 'BUS_1000', 'LAW_1000', 'SPO_1000'],
    #     '1B': ['GOV_1000', 'LIF_1000', 'HEA_1000', 'MIL_1000', 'SPO_1000']
    # }
    LANG_DIR = {
        '1A': ['GOV_1000', 'LIF_1000', ],
        '1B': ['GOV_1000', 'LIF_1000',  'MIL_1000', ]
    }

    for basedir in dirs:
        # TODO EVAL123
        if 'EVAL' not in basedir:
            continue
        if 'doc' not in basedir:
            continue
        lang = basedir[basedir.find('/') + 1:basedir.rfind('/')]
        for subdir in dirs[basedir]:
            # ignore bop
            if 'bop' in subdir:
                continue
            pred_dir = os.path.join(basedir, subdir)
            for dataset in LANG_DIR[lang]:
                for prefix in dataset_path_prefixes:
                    for architecture in architectures:
                        # print(dataset, prefix, seed, architecture, type, mode,                            pred_dir)
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

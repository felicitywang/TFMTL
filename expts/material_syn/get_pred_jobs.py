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

    # 'GOV_syn_1000',
    # 'LIF_syn_1000',
    # 'HEA_syn_1000',
    # 'LAW_syn_1000',
    # 'BUS_syn_1000',
    # 'MIL_syn_1000',
    # 'SPO_syn_1000',

    # 'TURK_GOV_90_50',
    # 'TURK_LIF_90_50',
    # 'TURK_BUS_90_50',
    # 'TURK_LAW_90_50',
    # 'TURK_HEA_90_50',
    # 'TURK_MIL_90_50',
    # 'TURK_SPO_90_50',

    # 'TURK_GOV_80_50',
    # 'TURK_LIF_80_50',
    # 'TURK_BUS_80_50',
    # 'TURK_LAW_80_50',
    # 'TURK_HEA_80_50',
    # 'TURK_MIL_80_50',
    # 'TURK_SPO_80_50',

    # 'TURK_GOV_70_50',
    # 'TURK_LIF_70_50',
    # 'TURK_BUS_70_50',
    # 'TURK_LAW_70_50',
    # 'TURK_HEA_70_50',
    # 'TURK_MIL_70_50',
    # 'TURK_SPO_70_50',

    # 'TURK_GOV_60_50',
    # 'TURK_LIF_60_50',
    # 'TURK_BUS_60_50',
    # 'TURK_LAW_60_50',
    # 'TURK_HEA_60_50',
    # 'TURK_MIL_60_50',
    # 'TURK_SPO_60_50',

    # 'TURK_GOV_50_50',
    # 'TURK_LIF_50_50',
    # 'TURK_BUS_50_50',
    # 'TURK_LAW_50_50',
    # 'TURK_MIL_50_50',
    # 'TURK_HEA_50_50',
    # 'TURK_SPO_50_50',

    # combined
    # 'GOV_syn_1000_TURK_90_50',
    # 'LIF_syn_1000_TURK_90_50',
    # 'BUS_syn_1000_TURK_90_50',
    # 'LAW_syn_1000_TURK_90_50',
    # 'HEA_syn_1000_TURK_90_50',
    # 'MIL_syn_1000_TURK_90_50',
    # 'SPO_syn_1000_TURK_90_50',
    # 'GOV_syn_1000_TURK_80_50',
    # 'LIF_syn_1000_TURK_80_50',
    # 'BUS_syn_1000_TURK_80_50',
    # 'LAW_syn_1000_TURK_80_50',
    # 'HEA_syn_1000_TURK_80_50',
    # 'MIL_syn_1000_TURK_80_50',
    # 'SPO_syn_1000_TURK_80_50',
    # 'GOV_syn_1000_TURK_70_50',
    # 'LIF_syn_1000_TURK_70_50',
    # 'BUS_syn_1000_TURK_70_50',
    # 'LAW_syn_1000_TURK_70_50',
    # 'HEA_syn_1000_TURK_70_50',
    # 'MIL_syn_1000_TURK_70_50',
    # 'SPO_syn_1000_TURK_70_50',
    # 'GOV_syn_1000_TURK_60_50',
    # 'LIF_syn_1000_TURK_60_50',
    # 'BUS_syn_1000_TURK_60_50',
    # 'LAW_syn_1000_TURK_60_50',
    # 'HEA_syn_1000_TURK_60_50',
    # 'MIL_syn_1000_TURK_60_50',
    # 'SPO_syn_1000_TURK_60_50',
    # 'GOV_syn_1000_TURK_50_50',
    # 'LIF_syn_1000_TURK_50_50',
    # 'BUS_syn_1000_TURK_50_50',
    # 'LAW_syn_1000_TURK_50_50',
    # 'HEA_syn_1000_TURK_50_50',
    # 'MIL_syn_1000_TURK_50_50',
    # 'SPO_syn_1000_TURK_50_50',

    # init
    'GOV_syn_1000_init_TURK_GOV_80_50',
    'LIF_syn_1000_init_TURK_LIF_80_50',
    'HEA_syn_1000_init_TURK_HEA_80_50',
    'MIL_syn_1000_init_TURK_MIL_80_50',
    'BUS_syn_1000_init_TURK_BUS_80_50',
    'LAW_syn_1000_init_TURK_LAW_80_50',
    'SPO_syn_1000_init_TURK_SPO_80_50',


]

CLASS_SIZE = 2


def make_job(dataset, prefix, seed, architecture, type, mode, pred_dir):
    dataset_paths = os.path.join('data/tf/single', dataset, prefix)
    alphas = '1'
    topic_paths = os.path.join('data/json/', dataset, 'data.json.gz')

    name = dataset + '_' + prefix + '_' + architecture + '_' + type

    domains = ['GOV', 'LIF', 'BUS', 'LAW', 'SPO', 'HEA', 'MIL']
    dataset_name = None
    for domain in domains:
        if domain in dataset:
            dataset_name = domain
    assert dataset_name is not None

    domains = ['GOV', 'LIF', 'BUS', 'LAW', 'SPO', 'HEA', 'MIL']
    dataset_name = None
    for domain in domains:
        if domain in dataset:
            dataset_name = domain
    assert dataset_name is not None

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

    if 'init' in dataset:
        predict_tfrecord = os.path.join('data/tf/predict/',
                                        pred_dir,
                                        dataset[:dataset.find('_init')],
                                        prefix,
                                        'predict.tf')

    else:
        predict_tfrecord = os.path.join('data/tf/predict/',
                                        pred_dir,
                                        dataset,
                                        prefix,
                                        'predict.tf')
    # predict_output_folder = os.path.join('predictions/', pred_dir,
    #                                      architecture + '_' + prefix)
    predict_output_folder = os.path.join('predictions/',
                                         pred_dir,
                                         dataset,
                                         prefix + '_' + architecture)
    make_dir(predict_output_folder)

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
cd /export/a08/fwang/tfmtl/expts/material_syn/\n" + \
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

        'meanmax_relu_nopretrain',
        # 'meanmax_relu_0.5_nopretrain',
        # 'meanmax_relu_0.1_nopretrain',

        # 'bilstm_nopretrain',

    ]

    # TODO
    LANG_DIR = {
        '1A': ['GOV', 'LIF', 'BUS', 'LAW', 'SPO'],
        '1B': ['GOV', 'LIF', 'HEA', 'MIL', 'SPO']
    }

    for basedir in dirs:
        # TODO EVAL123
        # dev only
        if 'DEV' not in basedir:
            continue
        # doc-level only
        if 'doc' not in basedir:
            continue
        lang = basedir[basedir.find('/') + 1:basedir.rfind('/')]
        for subdir in dirs[basedir]:
            # ignore bop
            if 'bop' in subdir:
                continue
            pred_dir = os.path.join(basedir, subdir)
            for dataset in TASKS:
                found = False
                for domain in LANG_DIR[lang]:
                    if domain in dataset:
                        found = True
                if not found:
                    continue
                for prefix in dataset_path_prefixes:
                    for architecture in architectures:
                        # print(dataset, prefix, seed, architecture, type, mode,                            pred_dir)
                        make_job(dataset, prefix, seed, architecture, type, mode,
                                 pred_dir)


if __name__ == "__main__":
    main()

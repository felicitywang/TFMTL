import os
import subprocess

from docutils.io import InputError

from mtl.util.util import make_dir

DEBUG = False

CLASS_SIZE = 2

METRICS = [
    'Acc',
    'F1_PosNeg_Macro',
    'Precision_Macro',
    'Recall_Macro',
    'Confusion_Matrix'
]


def make_job(dataset, prefix, seed, architecture, cpu_or_gpu, mode):
    dataset_paths = os.path.join('data/tf/single', dataset, prefix)
    alphas = '1'
    topic_paths = os.path.join('data/json/', dataset, 'data.json.gz')

    name = dataset + '_' + prefix + '_' + architecture + '_' + cpu_or_gpu

    domains = ['GOV', 'LIF', 'BUS', 'LAW', 'SPO', 'HEA', 'MIL']
    dataset_name = None
    for domain in domains:
        if domain in dataset:
            dataset_name = domain
    assert dataset_name is not None

    # name = dataset + '_' + prefix + '_' + architecture + \
    # '_gpu'  # Glove jobs are trained with GPU

    result_dir = os.path.join('results', 'seed_' + str(seed), name)
    make_dir(result_dir)

    if mode in ['train', 'test', 'predict', 'test_init']:
        checkpoint_dir = os.path.join(result_dir, 'ckpt')
    elif mode == 'init':
        checkpoint_dir = os.path.join(result_dir, 'ckpt')
        dataset_init = dataset[:dataset.find('_init')]
        print(dataset_init)
        name_init = dataset_init + '_' + prefix + '_' + architecture + '_' + cpu_or_gpu
        result_dir_init = os.path.join(
            'results', 'seed_' + str(seed), name_init)
        checkpoint_dir_init = os.path.join(result_dir_init, 'ckpt')
        # assert os.path.exists(checkpoint_dir_init)
    else:
        assert InputError('No such mode!')

    job_dir = os.path.join('jobs', 'seed_' + str(seed))
    make_dir(job_dir)
    job_name = mode + '_' + name + '.sh'
    job_path = os.path.join(job_dir, job_name)

    summ_dir = os.path.join('summ', 'seed_' + str(seed), name)
    make_dir(summ_dir)

    flag_l = {
        'cpu': "#$ -l \'mem_free=50G,ram_free=50G,hostname=\"b*|c*\"\'",
        'gpu': "#$ -l \'gpu=1,mem_free=50G,ram_free=50G,hostname=\"b1["
        "12345678]*|c*\" \n#$ -q g.q"
    }

    python = {
        'cpu': '/home/fwang/cpu/bin/python ',
        'gpu': 'CUDA_VISIBLE_DEVICES=`free-gpu` /home/fwang/gpu/bin/python '
    }

    encoder_path = 'encoders.json'

    if mode == 'test_init':
        mode = 'test'

    output = "#!/bin/sh\n\
#$ -cwd\n\
#$ -o " + os.path.join(result_dir, 'o') + "\n\
#$ -e " + os.path.join(result_dir, 'e') + "\n\
#$ -M cnfxwang@gmail.com\n" + \
             flag_l[cpu_or_gpu] + "\n\
#$ -V\n\
\n\
\n\
source /home/fwang/.bashrc\n\
cd /export/a08/fwang/tfmtl/expts/material_syn/\n" + \
             python[cpu_or_gpu] + \
             "../scripts/discriminative_driver.py \\\n" \
             "  --model mult \\\n" + \
             "  --num_train_epochs 30 \\\n" + \
             "  --optimizer rmsprop" + " \\\n" + \
             "  --lr0 0.001" + " \\\n" + \
             "  --patience 3" + " \\\n" + \
             "  --early_stopping_acc_threshold 1.0" + " \\\n" + \
             "  --shared_mlp_layers 0" + " \\\n" + \
             "  --shared_hidden_dims 0" + " \\\n" + \
             "  --private_mlp_layers 2" + " \\\n" + \
             "  --private_hidden_dims 128" + " \\\n" + \
             "  --input_keep_prob 1.0" + " \\\n" + \
             "  --output_keep_prob 0.5" + " \\\n" + \
             "  --l2_weight 0" + " \\\n" + \
             "  --tuning_metric Acc" + " \\\n" + \
             "  --checkpoint_dir " + checkpoint_dir + " " \
                                                      "\\" \
                                                      "\n" + \
             "  --mode " + mode + " \\\n" + \
             "  --summaries_dir " + summ_dir + " \\\n" + \
             "  --datasets " + dataset_name + " \\\n" + \
             "  --dataset_paths " + dataset_paths + " \\\n" + \
             "  --topics_paths " + topic_paths + " \\\n" + \
             "  --class_sizes " + str(CLASS_SIZE) + " \\\n" + \
             "  --encoder_config_file " + encoder_path + " \\\n" + \
             "  --architecture " + architecture + " \\\n" + \
             "  --seed " + str(seed) + " \\\n" + \
             "  --alphas " + alphas + " \\\n" + \
             "  --log_file " + os.path.join(result_dir, 'log') + " \\\n" + \
             "  --metrics " + ' '.join(METRICS)

    if mode == 'init':
        output += " \\\n  --checkpoint_dir_init " + checkpoint_dir_init

    output += "\n\n"

    print(output)

    with open(job_path, 'w') as file:
        file.write(output)

    if not DEBUG:
        subprocess.call(["qsub", job_path])


def main():
    dataset_path_prefixes = [
        'min_1_max_-1_vocab_-1_doc_-1',
        # 'min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand',
        # 'min_1_max_-1_vocab_-1_doc_1000_wiki-news-300d-1M_expand',
        # 'min_1_max_-1_vocab_-1_doc_1000_GoogleNews-vectors-negative300-SLIM_expand'
    ]

    import sys
    seed = sys.argv[1]
    cpu_or_gpu = sys.argv[2]
    mode = sys.argv[3]
    if len(sys.argv) != 4:
        raise InputError(
            'Usage: python get_jobs.py seed(integer) type(cpu/gpu) mode('
            'train/test/init)')

    architectures = [

        # 'meanmax_relu_nopretrain_0.0',
        'meanmax_relu_nopretrain',
        # 'meanmax_relu_0.5_nopretrain',
        # 'meanmax_relu_0.1_nopretrain',

        # 'bilstm_nopretrain',

        # 'bilstm_attention_3_nopretrain',

        # 'meanmax_linear_linear_nopretrain',
        # 'meanmax_linear_elu_nopretrain',
        # 'meanmax_linear_selu_nopretrain',
        # 'meanmax_linear_tanh_nopretrain',
        # 'meanmax_relu_relu_nopretrain',
        # 'meanmax_tanh_tanh_nopretrain',
        # 'meanmax_tanh_relu_nopretrain',
        # 'meanmax_relu_tanh_nopretrain',


    ]

    TASKS = [

        'GOV_syn_1000',
        'LIF_syn_1000',
        'HEA_syn_1000',
        'LAW_syn_1000',
        'BUS_syn_1000',
        'MIL_syn_1000',
        'SPO_syn_1000',

        'TURK_GOV_90_50',
        'TURK_LIF_90_50',
        'TURK_BUS_90_50',
        'TURK_LAW_90_50',
        'TURK_HEA_90_50',
        'TURK_MIL_90_50',
        'TURK_SPO_90_50',

        'TURK_GOV_80_50',
        'TURK_LIF_80_50',
        'TURK_BUS_80_50',
        'TURK_LAW_80_50',
        'TURK_HEA_80_50',
        'TURK_MIL_80_50',
        'TURK_SPO_80_50',

        'TURK_GOV_70_50',
        'TURK_LIF_70_50',
        'TURK_BUS_70_50',
        'TURK_LAW_70_50',
        'TURK_HEA_70_50',
        'TURK_MIL_70_50',
        'TURK_SPO_70_50',

        'TURK_GOV_60_50',
        'TURK_LIF_60_50',
        'TURK_BUS_60_50',
        'TURK_LAW_60_50',
        'TURK_HEA_60_50',
        'TURK_MIL_60_50',
        'TURK_SPO_60_50',

        'TURK_GOV_50_50',
        'TURK_LIF_50_50',
        'TURK_BUS_50_50',
        'TURK_LAW_50_50',
        'TURK_HEA_50_50',
        'TURK_MIL_50_50',
        'TURK_SPO_50_50',

        'GOV_syn_1000_TURK_90_50',
        'LIF_syn_1000_TURK_90_50',
        'BUS_syn_1000_TURK_90_50',
        'LAW_syn_1000_TURK_90_50',
        'HEA_syn_1000_TURK_90_50',
        'MIL_syn_1000_TURK_90_50',
        'SPO_syn_1000_TURK_90_50',
        'GOV_syn_1000_TURK_80_50',
        'LIF_syn_1000_TURK_80_50',
        'BUS_syn_1000_TURK_80_50',
        'LAW_syn_1000_TURK_80_50',
        'HEA_syn_1000_TURK_80_50',
        'MIL_syn_1000_TURK_80_50',
        'SPO_syn_1000_TURK_80_50',
        'GOV_syn_1000_TURK_70_50',
        'LIF_syn_1000_TURK_70_50',
        'BUS_syn_1000_TURK_70_50',
        'LAW_syn_1000_TURK_70_50',
        'HEA_syn_1000_TURK_70_50',
        'MIL_syn_1000_TURK_70_50',
        'SPO_syn_1000_TURK_70_50',
        'GOV_syn_1000_TURK_60_50',
        'LIF_syn_1000_TURK_60_50',
        'BUS_syn_1000_TURK_60_50',
        'LAW_syn_1000_TURK_60_50',
        'HEA_syn_1000_TURK_60_50',
        'MIL_syn_1000_TURK_60_50',
        'SPO_syn_1000_TURK_60_50',
        'GOV_syn_1000_TURK_50_50',
        'LIF_syn_1000_TURK_50_50',
        'BUS_syn_1000_TURK_50_50',
        'LAW_syn_1000_TURK_50_50',
        'HEA_syn_1000_TURK_50_50',
        'MIL_syn_1000_TURK_50_50',
        'SPO_syn_1000_TURK_50_50',


    ]

    # TODO
    if 'init' in mode:

        TASKS = [

            'GOV_syn_1000_init_TURK_GOV_70_50',
            'LIF_syn_1000_init_TURK_LIF_70_50',
            'HEA_syn_1000_init_TURK_HEA_70_50',
            'MIL_syn_1000_init_TURK_MIL_70_50',
            'BUS_syn_1000_init_TURK_BUS_70_50',
            'LAW_syn_1000_init_TURK_LAW_70_50',
            'SPO_syn_1000_init_TURK_SPO_70_50',

            'GOV_syn_1000_init_TURK_GOV_60_50',
            'LIF_syn_1000_init_TURK_LIF_60_50',
            'HEA_syn_1000_init_TURK_HEA_60_50',
            'MIL_syn_1000_init_TURK_MIL_60_50',
            'BUS_syn_1000_init_TURK_BUS_60_50',
            'LAW_syn_1000_init_TURK_LAW_60_50',
            'SPO_syn_1000_init_TURK_SPO_60_50',

            'GOV_syn_1000_init_TURK_GOV_50_50',
            'LIF_syn_1000_init_TURK_LIF_50_50',
            'HEA_syn_1000_init_TURK_HEA_50_50',
            'MIL_syn_1000_init_TURK_MIL_50_50',
            'BUS_syn_1000_init_TURK_BUS_50_50',
            'LAW_syn_1000_init_TURK_LAW_50_50',
            'SPO_syn_1000_init_TURK_SPO_50_50',

            'GOV_syn_1000_init_TURK_GOV_90_50',
            'LIF_syn_1000_init_TURK_LIF_90_50',
            'HEA_syn_1000_init_TURK_HEA_90_50',
            'MIL_syn_1000_init_TURK_MIL_90_50',
            'BUS_syn_1000_init_TURK_BUS_90_50',
            'LAW_syn_1000_init_TURK_LAW_90_50',
            'SPO_syn_1000_init_TURK_SPO_90_50',

        ]

    for prefix in dataset_path_prefixes:
        for architecture in architectures:
            for dataset in TASKS:
                make_job(dataset, prefix, seed, architecture, cpu_or_gpu, mode)


if __name__ == "__main__":
    main()

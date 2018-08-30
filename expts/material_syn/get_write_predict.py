"""Genearte bash scripts that write TFRecord files for files to predict

Example bash script:

python ../scripts/write_tfrecords_predict.py args_nopretrain.json data/json/predict/doc/1A/DEV/t6/mt-4.asr-s5/data.json.gz data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/GOV/min_1_max_-1_vocab_-1_doc_1000/predict.tf data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000/

Usage: python get_write_predict.py > predict.sh
"""

import os

from material_constants import PRED_DIRS as dirs

domains = ['GOV', 'LIF', 'BUS', 'LAW', 'HEA', 'MIL', 'SPO']
TASKS = [
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

    ]


def main():
    all_num = 0
    args_file = 'args_nopretrain.json'
    dataset_path = '/min_1_max_-1_vocab_-1_doc_-1/'
    # dataset_path = '/min_1_max_-1_vocab_-1_doc_1000/'
    # dataset_path = '/min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand'

    subdirs = {
        '1A': [
            't6/mt-4.asr-s5',
            'tt18',
            # 't6.bop/concat',
            # 'tt18.bop/concat'
        ],
        '1B': [
            't6/mt-5.asr-s5',
            'tt20',
            # 't6.bop/concat',
            # 'tt20.bop/concat'
        ]
    }

    EVAL_DIRS = [
        'DEV',
        # 'EVAL1',
        # 'EVAL2',
        # 'EVAL3',
        # 'ANALYSIS1',
        # 'ANALYSIS2',
        # 'goldDOMAIN'
    ]

    job_names = []

    tmp = {}
    for eval in EVAL_DIRS:
        # for doc_or_sent in ['doc', 'sent']:
        for doc_or_sent in ['doc']:
            for lang in ['1A', '1B']:
                # for lang in ['1A', '1B']:
                a = doc_or_sent + '/' + lang + '/' + eval
                if a in tmp:
                    tmp[a].append(subdirs[lang])
                else:
                    tmp[a] = subdirs[lang]
    # print(tmp)

    # for basedir in dirs:
    #     # TODO
    #     if 'sent' in basedir:
    #         continue
    #     # # TODO
    #     # if not 'DEV' in basedir and not 'EVAL' in basedir:
    #     if not 'DEV' in basedir:
    #         continue
    #     for subdir in dirs[basedir]:
    for basedir, subdirs in tmp.items():
        for subdir in subdirs:

            for task in TASKS:
                all_num += 1
                dir = os.path.join(basedir, subdir)
                # job = 'qsub -l mem_free=40G,ram_free=40G /home/fwang/cpu/bin/python ' \
                #       '/export/a08/fwang/tfmtl/expts/scripts' \
                #       '/write_tfrecords_predict.py ' \
                #       '/export/a08/fwang/tfmtl/expts/material_syn/' \
                #       + args_file + \
                #       ' /export/a08/fwang/tfmtl/expts/material_syn/data/json/predict/' + dir + '/data.json.gz' + \
                #       ' /export/a08/fwang/tfmtl/expts/material_syn/data/tf/predict/' + dir + '/' + task[:3] + dataset_path \
                #       + 'predict.tf' + \
                #       ' /export/a08/fwang/tfmtl/expts/material_syn/data/tf/single/' + task + dataset_path
                # for domain in domains:
                #   if domain in task:
                #     dataset_name = domain
                for domain in domains:
                    if domain in task:
                        dataset_name = domain
                dataset_tf_name = task
                # job = 'python ' \
                #       '../scripts/write_tfrecords_predict.py ' + args_file + \
                #       ' data/json/predict/' + dir + '/data.json.gz' + \
                #       ' data/tf/predict/' + dir + '/' + dataset_name + dataset_path \
                #       + 'predict.tf' + \
                #       ' data/tf/single/' + task + dataset_path
                # print(job)
                python = '/home/fwang/cpu/bin/python  '
                script = '/export/a08/fwang/tfmtl/expts/scripts' \
                         '/write_tfrecords_predict.py '
                args_file_path = '/export/a08/fwang/tfmtl/expts/' \
                                 '/material_syn/' + args_file + ' '
                json_path = '/export/a08/fwang/tfmtl/expts/material_syn/data/json' \
                            '/predict' \
                            '/' + dir + '/data.json.gz '
                tf_path = '/export/a08/fwang/tfmtl/expts/material_syn/data/tf' \
                          '/predict' \
                          '/' + dir + '/' + dataset_tf_name + dataset_path + 'predict.tf'
                arch_path = '/export/a08/fwang/tfmtl/expts/material_syn/data/tf/single/' + \
                            task + dataset_path

                # adafd
                # exit(-1)

                if os.path.exists(tf_path):
                    continue
                # else:
                #     print(tf_path)

                # print(task)
                # print(dataset_path)
                # print(arch_path)

                job = python + ' ' + script + ' ' + args_file_path + \
                    ' ' + json_path + ' ' + tf_path + ' ' + arch_path
                # print(job)
                #
                # job = 'python ' \
                #       '../scripts/write_tfrecords_predict.py ' + args_file + \
                #       ' data/json/predict/' + dir + '/data.json.gz' + \
                #       ' data/tf/predict/' + dir + '/' + dataset_name + dataset_path \
                #       + 'predict.tf' + \
                #       ' data/tf/single/' + task + dataset_path
                # print(job)

                job_name = dir + task + dataset_path

                import re
                job_name = re.sub('[/\- ._]', '', job_name) + '.sh'

                # print(job_name)
                job_names.append(job_name)

                with open(job_name, 'w') as fout:
                    fout.write(job + '\n')

                # print('qsub -l \'mem_free=4G,ram_free=4G,hostname=\"b*|c*\"\' -e '
                #       '/export/a08/fwang/tfmtl/expts/material_syn/write.e ', job_name)

                print(job)

    print(all_num)
    print(len(job_names))
    print(len(set(job_names)))


if __name__ == '__main__':
    main()

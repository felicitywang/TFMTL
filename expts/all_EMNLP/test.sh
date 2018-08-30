#!/bin/sh
#$ -cwd
#$ -o /export/a08/fwang/tfmtl/expts/all_EMNLP/o
#$ -e /export/a08/fwang/tfmtl/expts/all_EMNLP/e
#$ -M fwang40@jhu.edu
#$ -l 'gpu=1,mem_free=3G,ram_free=3G''
#$ -pe smp 2
#$ -V
#$ -q g.q

source /home/fwang/.bashrc


CUDA_VISIBLE_DEVICES=`free-gpu` /home/fwang/anaconda3/envs/gpu/bin/python3 /export/a08/fwang/tfmtl/expts/scripts/discriminative_driver.py \
--model mult \
--mode train \
--num_train_epochs 25 \
--optimizer rmsprop \
--lr0 0.001 \
--patience 3 \
--early_stopping_acc_threshold 1.0 \
--experiment_name RUDER_NAACL_18 \
--shared_mlp_layers 0 \
--shared_hidden_dims 0 \
--private_mlp_layers 1 \
--private_hidden_dims 100 \
--input_keep_prob 1.0 \
--output_keep_prob 1.0 \
--l2_weight 0 \
--tuning_metric Acc \
--checkpoint_dir /export/a08/fwang/tfmtl/expts/all_EMNLP/ckpt/ \
--datasets MultiNLI Topic5 \
--dataset_paths /export/a08/fwang/tfmtl/expts/all_EMNLP/data/tf/MultiNLI-mt/all_0_glove_init/MultiNLI /export/a08/fwang/tfmtl/expts/all_EMNLP/data/tf/MultiNLI-mt/all_0_glove_init/Topic5 \
--topics_paths /export/a08/fwang/tfmtl/expts/all_EMNLP/data/json/MultiNLI/data.json.gz /export/a08/fwang/tfmtl/expts/all_EMNLP/data/json/Topic5/data.json.gz \
--class_sizes 3 5 \
--vocab_size_file /export/a08/fwang/tfmtl/expts/all_EMNLP/data/tf/MultiNLI-mt/all_0_glove_init/vocab_size.txt \
--encoder_config_file encoder_files/encoder-mt-all_0.json \
--architecture serial_birnn_stock_glove_init_finetune \
--seed 42 \
--alphas 0.5 0.5 \
--log_file /export/a08/fwang/tfmtl/expts/all_EMNLP/log \
--reporting_metric Acc \

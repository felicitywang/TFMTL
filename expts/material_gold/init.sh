#!/bin/sh
#$ -cwd
#$ -o results/seed_42/GOV_50.0_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/GOV_50.0_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=20G,ram_free=20G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 30 \
  --optimizer rmsprop \
  --lr0 0.001 \
  --patience 3 \
  --early_stopping_acc_threshold 1.0 \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --tuning_metric Acc \
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu/ckpt \
  --mode init \
  --summaries_dir summ/seed_42/GOV_50.0_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu \
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_50.0/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/GOV_50.0/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/GOV_50.0_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/LIF_50.0_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/LIF_50.0_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=20G,ram_free=20G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 30 \
  --optimizer rmsprop \
  --lr0 0.001 \
  --patience 3 \
  --early_stopping_acc_threshold 1.0 \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --tuning_metric Acc \
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu/ckpt \
  --mode init \
  --summaries_dir summ/seed_42/LIF_50.0_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu \
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_50.0/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/LIF_50.0/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/LIF_50.0_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/HEA_50.0_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/HEA_50.0_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=20G,ram_free=20G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 30 \
  --optimizer rmsprop \
  --lr0 0.001 \
  --patience 3 \
  --early_stopping_acc_threshold 1.0 \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --tuning_metric Acc \
  --checkpoint_dir results/seed_42/HEA_1000_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu/ckpt \
  --mode init \
  --summaries_dir summ/seed_42/HEA_50.0_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu \
  --datasets HEA \
  --dataset_paths data/tf/single/HEA_50.0/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/HEA_50.0/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/HEA_50.0_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/MIL_50.0_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/MIL_50.0_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=20G,ram_free=20G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 30 \
  --optimizer rmsprop \
  --lr0 0.001 \
  --patience 3 \
  --early_stopping_acc_threshold 1.0 \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --tuning_metric Acc \
  --checkpoint_dir results/seed_42/MIL_1000_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu/ckpt \
  --mode init \
  --summaries_dir summ/seed_42/MIL_50.0_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu \
  --datasets MIL \
  --dataset_paths data/tf/single/MIL_50.0/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/MIL_50.0/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/MIL_50.0_min_1_max_-1_vocab_-1_doc_1000_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix



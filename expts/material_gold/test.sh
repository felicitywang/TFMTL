#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_GOV_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_GOV_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_GOV_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_GOV_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_GOV_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_LIF_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_LIF_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_LIF_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_LIF_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_LIF_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_BUS_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_BUS_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_BUS_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_BUS_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets BUS \
  --dataset_paths data/tf/single/TURK_BUS_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_BUS_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_BUS_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_LAW_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_LAW_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_LAW_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_LAW_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets LAW \
  --dataset_paths data/tf/single/TURK_LAW_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LAW_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_LAW_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_HEA_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_HEA_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_HEA_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_HEA_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets HEA \
  --dataset_paths data/tf/single/TURK_HEA_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_HEA_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_HEA_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_MIL_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_MIL_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_MIL_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_MIL_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets MIL \
  --dataset_paths data/tf/single/TURK_MIL_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_MIL_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_MIL_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_SPO_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_SPO_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_SPO_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_SPO_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_SPO_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_GOV_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_GOV_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_GOV_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_GOV_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_GOV_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_LIF_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_LIF_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_LIF_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_LIF_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_LIF_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_BUS_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_BUS_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_BUS_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_BUS_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets BUS \
  --dataset_paths data/tf/single/TURK_BUS_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_BUS_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_BUS_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_LAW_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_LAW_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_LAW_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_LAW_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets LAW \
  --dataset_paths data/tf/single/TURK_LAW_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LAW_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_LAW_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_HEA_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_HEA_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_HEA_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_HEA_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets HEA \
  --dataset_paths data/tf/single/TURK_HEA_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_HEA_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_HEA_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_MIL_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_MIL_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_MIL_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_MIL_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets MIL \
  --dataset_paths data/tf/single/TURK_MIL_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_MIL_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_MIL_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_SPO_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_SPO_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_SPO_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_SPO_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_SPO_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_GOV_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_GOV_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_GOV_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_GOV_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_GOV_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_LIF_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_LIF_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_LIF_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_LIF_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_LIF_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_BUS_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_BUS_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_BUS_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_BUS_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets BUS \
  --dataset_paths data/tf/single/TURK_BUS_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_BUS_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_BUS_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_LAW_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_LAW_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_LAW_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_LAW_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets LAW \
  --dataset_paths data/tf/single/TURK_LAW_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LAW_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_LAW_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_HEA_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_HEA_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_HEA_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_HEA_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets HEA \
  --dataset_paths data/tf/single/TURK_HEA_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_HEA_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_HEA_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_MIL_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_MIL_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_MIL_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_MIL_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets MIL \
  --dataset_paths data/tf/single/TURK_MIL_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_MIL_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_MIL_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_SPO_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_SPO_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_SPO_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_SPO_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_SPO_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_GOV_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_GOV_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_GOV_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_GOV_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_GOV_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_LIF_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_LIF_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_LIF_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_LIF_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_LIF_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_BUS_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_BUS_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_BUS_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_BUS_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets BUS \
  --dataset_paths data/tf/single/TURK_BUS_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_BUS_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_BUS_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_LAW_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_LAW_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_LAW_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_LAW_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets LAW \
  --dataset_paths data/tf/single/TURK_LAW_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LAW_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_LAW_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_HEA_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_HEA_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_HEA_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_HEA_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets HEA \
  --dataset_paths data/tf/single/TURK_HEA_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_HEA_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_HEA_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_MIL_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_MIL_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_MIL_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_MIL_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets MIL \
  --dataset_paths data/tf/single/TURK_MIL_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_MIL_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_MIL_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_SPO_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_SPO_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_SPO_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_SPO_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_SPO_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_GOV_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_GOV_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_GOV_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_GOV_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_GOV_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_LIF_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_LIF_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_LIF_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_LIF_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_LIF_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_BUS_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_BUS_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_BUS_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_BUS_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets BUS \
  --dataset_paths data/tf/single/TURK_BUS_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_BUS_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_BUS_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_LAW_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_LAW_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_LAW_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_LAW_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets LAW \
  --dataset_paths data/tf/single/TURK_LAW_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LAW_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_LAW_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_HEA_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_HEA_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_HEA_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_HEA_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets HEA \
  --dataset_paths data/tf/single/TURK_HEA_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_HEA_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_HEA_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_MIL_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_MIL_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_MIL_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_MIL_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets MIL \
  --dataset_paths data/tf/single/TURK_MIL_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_MIL_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_MIL_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/TURK_SPO_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/o
#$ -e results/seed_42/TURK_SPO_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 150 \
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
  --checkpoint_dir results/seed_42/TURK_SPO_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode test \
  --summaries_dir summ/seed_42/TURK_SPO_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu \
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/TURK_SPO_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix



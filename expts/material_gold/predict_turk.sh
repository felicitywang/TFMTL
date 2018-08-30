#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_GOV_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_LIF_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_BUS_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/TURK_BUS_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_BUS_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_BUS_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LAW_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/TURK_LAW_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LAW_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_LAW_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_SPO_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_GOV_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_LIF_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_BUS_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/TURK_BUS_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_BUS_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_BUS_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LAW_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/TURK_LAW_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LAW_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_LAW_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_SPO_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_GOV_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_LIF_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_BUS_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/TURK_BUS_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_BUS_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_BUS_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LAW_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/TURK_LAW_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LAW_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_LAW_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_SPO_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_GOV_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_LIF_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_BUS_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/TURK_BUS_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_BUS_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_BUS_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LAW_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/TURK_LAW_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LAW_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_LAW_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_SPO_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_GOV_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_LIF_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_BUS_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/TURK_BUS_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_BUS_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_BUS_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LAW_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/TURK_LAW_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LAW_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_LAW_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/TURK_SPO_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_GOV_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_LIF_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_BUS_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/TURK_BUS_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_BUS_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_BUS_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LAW_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/TURK_LAW_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LAW_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_LAW_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_SPO_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_GOV_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_LIF_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_BUS_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/TURK_BUS_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_BUS_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_BUS_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LAW_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/TURK_LAW_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LAW_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_LAW_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_SPO_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_GOV_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_LIF_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_BUS_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/TURK_BUS_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_BUS_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_BUS_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LAW_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/TURK_LAW_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LAW_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_LAW_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_SPO_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_GOV_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_LIF_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_BUS_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/TURK_BUS_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_BUS_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_BUS_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LAW_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/TURK_LAW_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LAW_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_LAW_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_SPO_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_GOV_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_LIF_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_BUS_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/TURK_BUS_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_BUS_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_BUS_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LAW_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/TURK_LAW_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LAW_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_LAW_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/TURK_SPO_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_GOV_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_LIF_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_HEA_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/TURK_HEA_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_HEA_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_HEA_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_MIL_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/TURK_MIL_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_MIL_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_MIL_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_SPO_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_GOV_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_LIF_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_HEA_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/TURK_HEA_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_HEA_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_HEA_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_MIL_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/TURK_MIL_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_MIL_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_MIL_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_SPO_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_GOV_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_LIF_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_HEA_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/TURK_HEA_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_HEA_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_HEA_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_MIL_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/TURK_MIL_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_MIL_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_MIL_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_SPO_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_GOV_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_LIF_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_HEA_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/TURK_HEA_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_HEA_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_HEA_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_MIL_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/TURK_MIL_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_MIL_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_MIL_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_SPO_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_GOV_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_LIF_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_HEA_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/TURK_HEA_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_HEA_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_HEA_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_MIL_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/TURK_MIL_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_MIL_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_MIL_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/TURK_SPO_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_GOV_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_LIF_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_HEA_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/TURK_HEA_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_HEA_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_HEA_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_MIL_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/TURK_MIL_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_MIL_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_MIL_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_90_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_SPO_90_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__90_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_GOV_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_LIF_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_HEA_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/TURK_HEA_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_HEA_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_HEA_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_MIL_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/TURK_MIL_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_MIL_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_MIL_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_80_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_SPO_80_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__80_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_GOV_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_LIF_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_HEA_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/TURK_HEA_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_HEA_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_HEA_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_MIL_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/TURK_MIL_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_MIL_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_MIL_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_70_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_SPO_70_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__70_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_GOV_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_LIF_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_HEA_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/TURK_HEA_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_HEA_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_HEA_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_MIL_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/TURK_MIL_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_MIL_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_MIL_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_60_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_SPO_60_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__60_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_GOV_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/TURK_GOV_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_GOV_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_GOV_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_LIF_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/TURK_LIF_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_LIF_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_LIF_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_HEA_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/TURK_HEA_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_HEA_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_HEA_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_MIL_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/TURK_MIL_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_MIL_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_MIL_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

#!/bin/sh
#$ -cwd
#$ -o pred.o
#$ -e pred.e
#$ -M cnfxwang@gmail.com
#$ -l 'mem_free=50G,ram_free=50G,hostname="b*|c*"'
#$ -pe smp 2
#$ -V


source /home/fwang/.bashrc
cd /export/a08/fwang/tfmtl/expts/material_gold/
/home/fwang/cpu/bin/python ../scripts/discriminative_driver.py \
  --model mult \
  --shared_mlp_layers 0 \
  --shared_hidden_dims 0 \
  --private_mlp_layers 2 \
  --private_hidden_dims 128 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/TURK_SPO_50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/TURK_SPO_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/TURK_SPO_50_50_ORACLE/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/TURK_SPO_50_50_ORACLE/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20/TURK__50_50_ORACLE_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain


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
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL1/t6/mt-4.asr-s5/GOV/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL1/t6/mt-4.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL1/t6/mt-4.asr-s5/LIF/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL1/t6/mt-4.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/BUS_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/BUS_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/BUS_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL1/t6/mt-4.asr-s5/BUS/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL1/t6/mt-4.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LAW_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/LAW_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LAW_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL1/t6/mt-4.asr-s5/LAW/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL1/t6/mt-4.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL1/t6/mt-4.asr-s5/SPO/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL1/t6/mt-4.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL1/tt18/GOV/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL1/tt18/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL1/tt18/LIF/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL1/tt18/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/BUS_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/BUS_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/BUS_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL1/tt18/BUS/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL1/tt18/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LAW_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/LAW_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LAW_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL1/tt18/LAW/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL1/tt18/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL1/tt18/SPO/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL1/tt18/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL1/t6/mt-5.asr-s5/GOV/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL1/t6/mt-5.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL1/t6/mt-5.asr-s5/LIF/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL1/t6/mt-5.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/HEA_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/HEA_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/HEA_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL1/t6/mt-5.asr-s5/HEA/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL1/t6/mt-5.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/MIL_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/MIL_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/MIL_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL1/t6/mt-5.asr-s5/MIL/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL1/t6/mt-5.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL1/t6/mt-5.asr-s5/SPO/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL1/t6/mt-5.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL1/tt20/GOV/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL1/tt20/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL1/tt20/LIF/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL1/tt20/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/HEA_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/HEA_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/HEA_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL1/tt20/HEA/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL1/tt20/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/MIL_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/MIL_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/MIL_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL1/tt20/MIL/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL1/tt20/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL1/tt20/SPO/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL1/tt20/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL2/t6/mt-4.asr-s5/GOV/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL2/t6/mt-4.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL2/t6/mt-4.asr-s5/LIF/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL2/t6/mt-4.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/BUS_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/BUS_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/BUS_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL2/t6/mt-4.asr-s5/BUS/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL2/t6/mt-4.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LAW_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/LAW_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LAW_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL2/t6/mt-4.asr-s5/LAW/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL2/t6/mt-4.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL2/t6/mt-4.asr-s5/SPO/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL2/t6/mt-4.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL2/tt18/GOV/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL2/tt18/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL2/tt18/LIF/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL2/tt18/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/BUS_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/BUS_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/BUS_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL2/tt18/BUS/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL2/tt18/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LAW_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/LAW_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LAW_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL2/tt18/LAW/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL2/tt18/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL2/tt18/SPO/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL2/tt18/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL2/t6/mt-5.asr-s5/GOV/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL2/t6/mt-5.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL2/t6/mt-5.asr-s5/LIF/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL2/t6/mt-5.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/HEA_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/HEA_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/HEA_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL2/t6/mt-5.asr-s5/HEA/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL2/t6/mt-5.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/MIL_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/MIL_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/MIL_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL2/t6/mt-5.asr-s5/MIL/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL2/t6/mt-5.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL2/t6/mt-5.asr-s5/SPO/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL2/t6/mt-5.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL2/tt20/GOV/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL2/tt20/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL2/tt20/LIF/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL2/tt20/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/HEA_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/HEA_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/HEA_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL2/tt20/HEA/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL2/tt20/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/MIL_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/MIL_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/MIL_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL2/tt20/MIL/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL2/tt20/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL2/tt20/SPO/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL2/tt20/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL3/t6/mt-4.asr-s5/GOV/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL3/t6/mt-4.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL3/t6/mt-4.asr-s5/LIF/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL3/t6/mt-4.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/BUS_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/BUS_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/BUS_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL3/t6/mt-4.asr-s5/BUS/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL3/t6/mt-4.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LAW_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/LAW_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LAW_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL3/t6/mt-4.asr-s5/LAW/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL3/t6/mt-4.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL3/t6/mt-4.asr-s5/SPO/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL3/t6/mt-4.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL3/tt18/GOV/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL3/tt18/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL3/tt18/LIF/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL3/tt18/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/BUS_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/BUS_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/BUS_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL3/tt18/BUS/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL3/tt18/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LAW_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/LAW_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LAW_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL3/tt18/LAW/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL3/tt18/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/EVAL3/tt18/SPO/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1A/EVAL3/tt18/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL3/t6/mt-5.asr-s5/GOV/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL3/t6/mt-5.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL3/t6/mt-5.asr-s5/LIF/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL3/t6/mt-5.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/HEA_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/HEA_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/HEA_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL3/t6/mt-5.asr-s5/HEA/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL3/t6/mt-5.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/MIL_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/MIL_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/MIL_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL3/t6/mt-5.asr-s5/MIL/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL3/t6/mt-5.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL3/t6/mt-5.asr-s5/SPO/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL3/t6/mt-5.asr-s5/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL3/tt20/GOV/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL3/tt20/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL3/tt20/LIF/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL3/tt20/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/HEA_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/HEA_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/HEA_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL3/tt20/HEA/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL3/tt20/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/MIL_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/MIL_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/MIL_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL3/tt20/MIL/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL3/tt20/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain

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
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_-1 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --encoder_config_file encoders.json \
  --architecture meanmax_relu_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/EVAL3/tt20/SPO/min_1_max_-1_vocab_-1_doc_-1/predict.tf \
  --predict_output_folder predictions/doc/1B/EVAL3/tt20/min_1_max_-1_vocab_-1_doc_-1_meanmax_relu_nopretrain


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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/GOV/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/LIF/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/BUS_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/BUS_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/BUS_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/BUS_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/BUS/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/LAW_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/LAW_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/LAW_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/LAW_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/LAW/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6/mt-4.asr-s5/SPO/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6/mt-4.asr-s5 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/GOV/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/LIF/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/BUS_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/BUS_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/BUS_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/BUS_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/BUS/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/LAW_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/LAW_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/LAW_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/LAW_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/LAW/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18/SPO/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6.bop/concat/GOV/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6.bop/concat/LIF/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/BUS_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/BUS_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/BUS_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/BUS_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6.bop/concat/BUS/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/LAW_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/LAW_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/LAW_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/LAW_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6.bop/concat/LAW/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/t6.bop/concat/SPO/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/t6.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18.bop/concat/GOV/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18.bop/concat/LIF/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/BUS_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets BUS \
  --dataset_paths data/tf/single/BUS_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/BUS_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/BUS_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset BUS \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18.bop/concat/BUS/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/LAW_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LAW \
  --dataset_paths data/tf/single/LAW_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/LAW_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/LAW_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LAW \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18.bop/concat/LAW/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1A/DEV/tt18.bop/concat/SPO/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1A/DEV/tt18.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/GOV/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/LIF/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/HEA_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/HEA_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/HEA_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/HEA_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/HEA/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/MIL_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/MIL_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/MIL_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/MIL_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/MIL/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6/mt-5.asr-s5/SPO/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6/mt-5.asr-s5 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/GOV/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/LIF/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/HEA_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/HEA_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/HEA_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/HEA_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/HEA/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/MIL_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/MIL_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/MIL_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/MIL_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/MIL/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20/SPO/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6.bop/concat/GOV/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6.bop/concat/LIF/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/HEA_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/HEA_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/HEA_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/HEA_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6.bop/concat/HEA/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/MIL_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/MIL_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/MIL_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/MIL_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6.bop/concat/MIL/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/t6.bop/concat/SPO/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/t6.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset GOV \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20.bop/concat/GOV/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset LIF \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20.bop/concat/LIF/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/HEA_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets HEA \
  --dataset_paths data/tf/single/HEA_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/HEA_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/HEA_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset HEA \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20.bop/concat/HEA/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/MIL_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets MIL \
  --dataset_paths data/tf/single/MIL_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/MIL_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/MIL_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset MIL \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20.bop/concat/MIL/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20.bop/concat 

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
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --checkpoint_dir results/seed_42/SPO_1000_min_1_max_-1_vocab_-1_doc_1000_paragram_nopretrain_cpu/ckpt \
  --mode predict\
  --datasets SPO \
  --dataset_paths data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_1000 \
  --topics_paths data/json/SPO_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/SPO_1000/min_1_max_-1_vocab_-1_doc_1000/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_nopretrain \
  --seed 42 \
  --alphas 1 \
  --predict_dataset SPO \
  --predict_tfrecord data/tf/predict/doc/1B/DEV/tt20.bop/concat/SPO/min_1_max_-1_vocab_-1_doc_1000/predict.tf \
  --predict_output_folder predictions/doc/1B/DEV/tt20.bop/concat 


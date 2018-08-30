#!/bin/sh

source /home/fwang/.bashrc
source /home/fwang/gpu/bin/activate
pip list | grep tensorflow
python -c 'import tensorflow as tf; print(tf.__version__)' 
CUDA_VISIBLE_DEVICES=`free-gpu` python ../scripts/discriminative_driver.py \
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
    --checkpoint_dir ./ckpt/ \
    --datasets MultiNLI Topic5 \
    --dataset_paths ./data/tf/MultiNLI-mt/train_1_glove_init/MultiNLI ./data/tf/MultiNLI-mt/train_1_glove_init/Topic5 \
    --topics_paths ./data/json/MultiNLI/data.json.gz ./data/json/Topic5/data.json.gz \
    --class_sizes 3 5 \
    --vocab_size_file ./data/tf/MultiNLI-mt/train_1_glove_init/vocab_size.txt \
    --encoder_config_file encoder_files/encoder-mt-train_1.json \
    --architecture serial_birnn_stock_glove_init_finetune \
    --seed 42 \
    --alphas 0.5 0.5 \
    --log_file ./log \
    --reporting_metric Acc \

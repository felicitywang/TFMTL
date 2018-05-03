python ../scripts/discriminative_driver.py \
       --model mult \
       --mode train \
       --num_train_epochs 50 \
       --checkpoint_dir ./data/ckpt/mult_glove_and_train/ \
       --experiment_name RUDER_NAACL_18 \
       --datasets Target Topic2\
       --dataset_paths \
       data/tf/merged/Target_Topic2/glove.6B.100d_and_train/Target \
       data/tf/merged/Target_Topic2/glove.6B.100d_and_train/Topic2 \
       --class_sizes 3 2 \
       --vocab_size_file data/tf/merged/Target_Topic2/glove.6B.100d_and_train/vocab_size.txt \
       --encoder_config_file encoders.json \
       --architecture serial-birnn-glove-and-train \
       --shared_mlp_layers 0 \
       --shared_hidden_dims 0 \
       --private_mlp_layers 1 \
       --private_hidden_dims 64 \
       --alphas 0.5 0.5 \
       --optimizer rmsprop \
       --lr0 0.001 \
       --tuning_metric Acc \
       --topics_path \
       data/json/Target/data.json.gz \
       data/json/Topic2/data.json.gz \
       --seed 42 \
       --log_file mult_and_train.log
python ../scripts/discriminative_driver.py \
    --model mult \
    --mode test \
    --num_train_epochs 30 \
    --experiment_name RUDER_NAACL_18 \
    --datasets Topic2 \
    --class_sizes 2 \
    --dataset_paths data/tf/single/Topic2/min_1_max_-1_vocab_-1_doc_-1_tok_tweet/ \
    --topics_path data/json/Topic2/data.json.gz \
    --topic_field_name seq1 \
    --encoder_config_file encoders.json \
    --architecture meanmax_relu_0.1_nopretrain \
    --shared_mlp_layers 0 \
    --shared_hidden_dims 0 \
    --private_mlp_layers 1 \
    --private_hidden_dims 64 \
    --alphas 1 \
    --optimizer rmsprop \
    --lr0 0.001 \
    --seed 42 \
    --summaries_dir ./data/summ/Topic2_nopretrain/ \
    --checkpoint_dir ./data/ckpt/Topic2_nopretrain/ \
    --log_file ./data/logs/Topic2_nopretrain.log \
    --tuning_metric Acc \

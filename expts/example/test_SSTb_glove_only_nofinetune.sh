python ../scripts/discriminative_driver.py \
       --model mult \
       --mode test \
       --num_train_epochs 30 \
       --datasets SSTb \
       --class_sizes 5 \
       --dataset_paths data/tf/single/SSTb/min_1_max_-1_vocab_-1_doc_-1_tok_lower_glove.6B.100d_only/ \
       --topics_path data/json/SSTb/data.json.gz \
       --topic_field_name text \
       --encoder_config_file encoders.json \
       --architecture meanmax_relu_0.1_glove_only_nofinetune \
       --shared_mlp_layers 0 \
       --shared_hidden_dims 0 \
       --private_mlp_layers 1 \
       --private_hidden_dims 64 \
       --alphas 1 \
       --optimizer rmsprop \
       --lr0 0.001 \
       --seed 42 \
       --summaries_dir ./data/summ/SSTb_glove_only_nofinetune/ \
       --checkpoint_dir ./data/ckpt/SSTb_glove_only_nofinetune/ \
       --log_file ./data/logs/SSTb_glove_only_nofinetune.log \
       --tuning_metric Acc \
       --metrics Acc Precision_Macro Recall_Macro F1_Macro
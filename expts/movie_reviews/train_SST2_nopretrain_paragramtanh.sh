python ../scripts/discriminative_driver.py \
       --model mult \
       --mode train \
       --num_train_epochs 30 \
       --datasets SST2 \
       --class_sizes 5 \
       --dataset_paths data/tf/single/SST2/min_1_max_-1_vocab_-1_doc_-1_tok_ruder/ \
       --topics_path data/json/SST2/data.json.gz \
       --topic_field_name text \
       --encoder_config_file encoders.json \
       --architecture meanmax_relu_0.1_nopretrain \
       --shared_mlp_layers 0 \
       --shared_hidden_dims 0 \
       --private_mlp_layers 1 \
       --private_hidden_dims 128 \
       --alphas 1 \
       --optimizer rmsprop \
       --lr0 0.001 \
       --seed 42 \
       --summaries_dir ./data/summ/SST2_nopretrain/ \
       --checkpoint_dir ./data/ckpt/SST2_nopretrain/ \
       --log_file ./data/logs/SST2_nopretrain.log \
       --tuning_metric F1_Macro \
       --reporting_metric F1_Macro \
       --metrics Acc Precision_Macro Recall_Macro F1_Macro

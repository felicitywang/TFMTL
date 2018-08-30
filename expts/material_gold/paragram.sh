python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 30 \
  --optimizer rmsprop \
  --lr0 0.001 \
  --patience 3 \
  --early_stopping_acc_threshold 1.0 \
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --tuning_metric Acc \
  --mode test\
  --checkpoint_dir results/seed_42/GOV_1000_min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand_paragram_expand_glove_cpu/ckpt \
  --datasets GOV \
  --dataset_paths data/tf/pilot_1_synthetic/GOV_50.0/min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand \
  --topics_paths data/json/GOV_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_expand_glove \
  --seed 42 \
  --alphas 1 \
  --log_file paragram.log \
  --metrics Acc F1_PosNeg_Macro 


python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 30 \
  --optimizer rmsprop \
  --lr0 0.001 \
  --patience 3 \
  --early_stopping_acc_threshold 1.0 \
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --tuning_metric Acc \
  --mode test\
  --checkpoint_dir results/seed_42/LIF_1000_min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand_paragram_expand_glove_cpu/ckpt \
  --datasets LIF \
  --dataset_paths data/tf/pilot_1_synthetic/LIF_50.0/min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand \
  --topics_paths data/json/LIF_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/LIF_1000/min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_expand_glove \
  --seed 42 \
  --alphas 1 \
  --log_file paragram.log \
  --metrics Acc F1_PosNeg_Macro 


python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 30 \
  --optimizer rmsprop \
  --lr0 0.001 \
  --patience 3 \
  --early_stopping_acc_threshold 1.0 \
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --tuning_metric Acc \
  --mode test\
  --checkpoint_dir results/seed_42/HEA_1000_min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand_paragram_expand_glove_cpu/ckpt \
  --datasets HEA \
  --dataset_paths data/tf/pilot_1_synthetic/HEA_50.0/min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand \
  --topics_paths data/json/HEA_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/HEA_1000/min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_expand_glove \
  --seed 42 \
  --alphas 1 \
  --log_file paragram.log \
  --metrics Acc F1_PosNeg_Macro 


python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 30 \
  --optimizer rmsprop \
  --lr0 0.001 \
  --patience 3 \
  --early_stopping_acc_threshold 1.0 \
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --tuning_metric Acc \
  --mode test\
  --checkpoint_dir results/seed_42/LAW_1000_min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand_paragram_expand_glove_cpu/ckpt \
  --datasets LAW \
  --dataset_paths data/tf/pilot_1_synthetic/LAW_50.0/min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand \
  --topics_paths data/json/LAW_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/LAW_1000/min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_expand_glove \
  --seed 42 \
  --alphas 1 \
  --log_file paragram.log \
  --metrics Acc F1_PosNeg_Macro 


python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 30 \
  --optimizer rmsprop \
  --lr0 0.001 \
  --patience 3 \
  --early_stopping_acc_threshold 1.0 \
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --tuning_metric Acc \
  --mode test\
  --checkpoint_dir results/seed_42/BUS_1000_min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand_paragram_expand_glove_cpu/ckpt \
  --datasets BUS \
  --dataset_paths data/tf/pilot_1_synthetic/BUS_50.0/min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand \
  --topics_paths data/json/BUS_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/BUS_1000/min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_expand_glove \
  --seed 42 \
  --alphas 1 \
  --log_file paragram.log \
  --metrics Acc F1_PosNeg_Macro 


python ../scripts/discriminative_driver.py \
  --model mult \
  --num_train_epochs 30 \
  --optimizer rmsprop \
  --lr0 0.001 \
  --patience 3 \
  --early_stopping_acc_threshold 1.0 \
  --shared_mlp_layers 1 \
  --shared_hidden_dims 100 \
  --private_mlp_layers 1 \
  --private_hidden_dims 100 \
  --input_keep_prob 1.0 \
  --output_keep_prob 0.5 \
  --l2_weight 0 \
  --tuning_metric Acc \
  --mode test\
  --checkpoint_dir results/seed_42/MIL_1000_min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand_paragram_expand_glove_cpu/ckpt \
  --datasets MIL \
  --dataset_paths data/tf/pilot_1_synthetic/MIL_50.0/min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand \
  --topics_paths data/json/MIL_1000/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/MIL_1000/min_1_max_-1_vocab_-1_doc_1000_glove.6B.300d_expand/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture paragram_expand_glove \
  --seed 42 \
  --alphas 1 \
  --log_file paragram.log \
  --metrics Acc F1_PosNeg_Macro 



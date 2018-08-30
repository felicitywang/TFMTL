Usage: python get_jobs.py seed(integer) type(cpu/gpu) mode(train/test)
#!/bin/sh
#$ -cwd
#$ -o results/seed_42/GOV_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/o
#$ -e results/seed_42/GOV_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'gpu=1,mem_free=20G,ram_free=20G,hostname="b1[12345678]*|c*" 
#$ -q g.q
#$ -V


source /home/fwang/.bashrc 
cd /export/a08/fwang/tfmtl/expts/material_gold/ 
CUDA_VISIBLE_DEVICES=`free-gpu` /home/fwang/gpu/bin/python ../scripts/discriminative_driver.py \
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
  --checkpoint_dir results/seed_42/GOV_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/ckpt \
  --mode train\
  --summaries_dir summ/seed_42/GOV_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu \
  --datasets GOV \
  --dataset_paths data/tf/single/GOV_100/min_1_max_-1_vocab_-1_doc_400 \
  --topics_paths data/json/GOV_100/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/GOV_100/min_1_max_-1_vocab_-1_doc_400/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture bilstm_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/GOV_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix 


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/LIF_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/o
#$ -e results/seed_42/LIF_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'gpu=1,mem_free=20G,ram_free=20G,hostname="b1[12345678]*|c*" 
#$ -q g.q
#$ -V


source /home/fwang/.bashrc 
cd /export/a08/fwang/tfmtl/expts/material_gold/ 
CUDA_VISIBLE_DEVICES=`free-gpu` /home/fwang/gpu/bin/python ../scripts/discriminative_driver.py \
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
  --checkpoint_dir results/seed_42/LIF_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/ckpt \
  --mode train\
  --summaries_dir summ/seed_42/LIF_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu \
  --datasets LIF \
  --dataset_paths data/tf/single/LIF_100/min_1_max_-1_vocab_-1_doc_400 \
  --topics_paths data/json/LIF_100/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/LIF_100/min_1_max_-1_vocab_-1_doc_400/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture bilstm_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/LIF_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix 


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/HEA_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/o
#$ -e results/seed_42/HEA_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'gpu=1,mem_free=20G,ram_free=20G,hostname="b1[12345678]*|c*" 
#$ -q g.q
#$ -V


source /home/fwang/.bashrc 
cd /export/a08/fwang/tfmtl/expts/material_gold/ 
CUDA_VISIBLE_DEVICES=`free-gpu` /home/fwang/gpu/bin/python ../scripts/discriminative_driver.py \
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
  --checkpoint_dir results/seed_42/HEA_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/ckpt \
  --mode train\
  --summaries_dir summ/seed_42/HEA_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu \
  --datasets HEA \
  --dataset_paths data/tf/single/HEA_100/min_1_max_-1_vocab_-1_doc_400 \
  --topics_paths data/json/HEA_100/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/HEA_100/min_1_max_-1_vocab_-1_doc_400/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture bilstm_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/HEA_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix 


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/LAW_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/o
#$ -e results/seed_42/LAW_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'gpu=1,mem_free=20G,ram_free=20G,hostname="b1[12345678]*|c*" 
#$ -q g.q
#$ -V


source /home/fwang/.bashrc 
cd /export/a08/fwang/tfmtl/expts/material_gold/ 
CUDA_VISIBLE_DEVICES=`free-gpu` /home/fwang/gpu/bin/python ../scripts/discriminative_driver.py \
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
  --checkpoint_dir results/seed_42/LAW_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/ckpt \
  --mode train\
  --summaries_dir summ/seed_42/LAW_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu \
  --datasets LAW \
  --dataset_paths data/tf/single/LAW_100/min_1_max_-1_vocab_-1_doc_400 \
  --topics_paths data/json/LAW_100/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/LAW_100/min_1_max_-1_vocab_-1_doc_400/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture bilstm_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/LAW_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix 


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/BUS_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/o
#$ -e results/seed_42/BUS_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'gpu=1,mem_free=20G,ram_free=20G,hostname="b1[12345678]*|c*" 
#$ -q g.q
#$ -V


source /home/fwang/.bashrc 
cd /export/a08/fwang/tfmtl/expts/material_gold/ 
CUDA_VISIBLE_DEVICES=`free-gpu` /home/fwang/gpu/bin/python ../scripts/discriminative_driver.py \
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
  --checkpoint_dir results/seed_42/BUS_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/ckpt \
  --mode train\
  --summaries_dir summ/seed_42/BUS_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu \
  --datasets BUS \
  --dataset_paths data/tf/single/BUS_100/min_1_max_-1_vocab_-1_doc_400 \
  --topics_paths data/json/BUS_100/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/BUS_100/min_1_max_-1_vocab_-1_doc_400/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture bilstm_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/BUS_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix 


#!/bin/sh
#$ -cwd
#$ -o results/seed_42/MIL_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/o
#$ -e results/seed_42/MIL_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/e
#$ -M cnfxwang@gmail.com
#$ -l 'gpu=1,mem_free=20G,ram_free=20G,hostname="b1[12345678]*|c*" 
#$ -q g.q
#$ -V


source /home/fwang/.bashrc 
cd /export/a08/fwang/tfmtl/expts/material_gold/ 
CUDA_VISIBLE_DEVICES=`free-gpu` /home/fwang/gpu/bin/python ../scripts/discriminative_driver.py \
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
  --checkpoint_dir results/seed_42/MIL_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/ckpt \
  --mode train\
  --summaries_dir summ/seed_42/MIL_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu \
  --datasets MIL \
  --dataset_paths data/tf/single/MIL_100/min_1_max_-1_vocab_-1_doc_400 \
  --topics_paths data/json/MIL_100/data.json.gz \
  --class_sizes 2 \
  --vocab_size_file data/tf/single/MIL_100/min_1_max_-1_vocab_-1_doc_400/vocab_size.txt \
  --encoder_config_file encoders.json \
  --architecture bilstm_nopretrain \
  --seed 42 \
  --alphas 1 \
  --log_file results/seed_42/MIL_100_min_1_max_-1_vocab_-1_doc_400_bilstm_nopretrain_gpu/log \
  --metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix 



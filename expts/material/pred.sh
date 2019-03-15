source /home/fwang/cpu/bin/activate
cd /export/a08/fwang/tfmtl/expts/material
python /export/a08/fwang/tfmtl/expts/scripts/discriminative_driver.py \
    --private_hidden_dims 128  \
    --class_sizes 2  \
    --metrics Acc Precision_Macro Recall_Macro F1_Macro  \
    --tuning_metric Acc  \
    --shared_mlp_layers 0  \
    --predict_dataset REL  \
    --optimizer rmsprop  \
    --dataset_paths /export/a08/fwang/tfmtl/expts/material/data/tf/single/REL_syn_p1000r1000/min_1_max_-1_vocab_-1_doc_-1_tok_lower  \
    --log_file /export/a08/fwang/tfmtl/expts/material/data/results/REL_syn_p1000r1000_min_1_max_-1_vocab_-1_doc_-1_tok_lower_dan_meanmax_relu_0.1_nopretrain/predict.log  \
    --architecture dan_meanmax_relu_0.1_nopretrain  \
    --encoder_config_file /export/a08/fwang/tfmtl/expts/material/data/results/REL_syn_p1000r1000_min_1_max_-1_vocab_-1_doc_-1_tok_lower_dan_meanmax_relu_0.1_nopretrain/encoders.json  \
    --predict_output_folder /export/a08/fwang/tfmtl/expts/material/data/predictions/sent/REL/REL_syn_p1000r1000_min_1_max_-1_vocab_-1_doc_-1_tok_lower_dan_meanmax_relu_0.1_nopretrain  \
    --l2_weight 0.0  \
    --topics_paths /export/a08/fwang/tfmtl/expts/material/data/json/REL_syn_p1000r1000  \
    --private_mlp_layers 1  \
    --num_intra_threads 1  \
    --input_keep_prob 1.0  \
    --num_inter_threads 1  \
    --alphas 1  \
    --output_keep_prob 1.0  \
    --shared_hidden_dims 0  \
    --seed 42  \
    --num_train_epochs 30  \
    --lr0 0.001  \
    --predict_tfrecord_path /export/a08/fwang/tfmtl/expts/material/data/pred/tf/sent/REL/pred.tf  \
    --datasets REL  \
    --checkpoint_dir /export/a08/fwang/tfmtl/expts/material/data/results/REL_syn_p1000r1000_min_1_max_-1_vocab_-1_doc_-1_tok_lower_dan_meanmax_relu_0.1_nopretrain/ckpt  \
    --model mult  \
    --mode predict  \
    --early_stopping_acc_threshold 1.0 


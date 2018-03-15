python ../scripts/discriminative_driver.py --datasets LMRD --dataset_paths data/tf/single/LMRD/min_50_max_-1/ --class_sizes 2 --vocab_path data/tf/single/LMRD/min_50_max_-1/vocab_size.txt --architectures_path architectures.json --model mult --alphas 1 --encoder_architecture paragram_phrase_tied_word_embeddings --mode train --num_train_epochs 3 --checkpoint_dir ./data/ckpt/LMRD/



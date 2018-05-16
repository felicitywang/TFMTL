python ../scripts/discriminative_driver.py --datasets LMRD --dataset_paths data/tf/single/LMRD/min_0_max_-1_vocab_10000/ --class_sizes 2 --vocab_path data/tf/single/LMRD/min_0_max_-1_vocab_10000/vocab_size.txt --encoder_config_file encoders.json --model mult --alphas 1 --architecture paragram --mode test --checkpoint_dir ./data/ckpt/LMRD/


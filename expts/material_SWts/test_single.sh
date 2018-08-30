python ../scripts/discriminative_driver.py --datasets SWts --dataset_paths data/tf/single/SWts/min_0_max_-1_vocab_10000/ --class_sizes 2 --vocab_size_file data/tf/single/SWts/min_0_max_-1_vocab_10000/vocab_size.txt --encoder_config_file encoders.json --model mult --alphas 1 --architecture paragram --mode test --checkpoint_dir ./data/ckpt/single/SWts/


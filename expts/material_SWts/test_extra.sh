# test extra SWts with single
python ../scripts/discriminative_driver.py --datasets SWts --dataset_paths data/tf/extra_single/ --class_sizes 2 --vocab_size_file data/tf/single/SWts/min_0_max_-1_vocab_10000/vocab_size.txt --encoder_config_file encoders.json --model mult --alphas 1 --architecture paragram --mode test --checkpoint_dir ./data/ckpt/single/SWts/

# test extra SWts with mult
python ../scripts/discriminative_driver.py --datasets SWts --dataset_paths data/tf/extra_mult/ --class_sizes 2 --vocab_size_file data/tf/single/SWts/min_0_max_-1_vocab_10000/vocab_size.txt --encoder_config_file encoders.json --model mult --alphas 1 --architecture paragram --mode test --checkpoint_dir ./data/ckpt/mult/LMRD_SWts/

# test extra SWts + prev LMRD with mult
python ../scripts/discriminative_driver.py --datasets LMRD SWts --dataset_paths data/tf/merged/LMRD_SWts/min_0_max_-1_vocab_10000/LMRD/ data/tf/extra_mult/ --class_sizes 2 2 --vocab_size_file data/tf/merged/LMRD_SWts/min_0_max_-1_vocab_10000/vocab_size.txt --encoder_config_file encoders.json --model mult --input_key tokens --architecture paragram --alphas 0.5 0.5 --mode test --checkpoint_dir ./data/ckpt/mult/LMRD_SWts/


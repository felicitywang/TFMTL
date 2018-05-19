# Steps

1. Run `./setup.sh` to download all the data and convert them to json format. Json files would be saved in `data/json/`.
2. Modify `dataset` and `max_len` and run `python truncate_examples.py` to truncate the FNC-1/MultiNLI json file in place to the maximum length.
3. Run `python write.py TARGET_DATASET` to write TFRecord files for the target dataset with its auxiliary datasets in table 4 of the Ruder paper. Data files would be originally saved to `data/tf/single/DATASET_NAME/vocab_args/` or `data/tf/merged/sorted_DATASET_NAMES/vocab_args/DATASET_NAME/`. It will be moved into `data/tf/TARGET_DATASET_NAME-st/vocab_mode/DATASET_NAME/`. The paths in `args.json` would be the original path. Original paths would then be soft linked to the current path to make sure the saved paths can be used.

## vocab mode:
- all_0(Ruder vocab): using all train/dev/test splits for vocabulary, minimum frequency = 0(using all the words in the whole dataset)
- train_1(our vocab): using the train split with minimum frequency = 1


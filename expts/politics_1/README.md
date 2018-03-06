## datasets

type|name|#items|#labels|unit|summary|split|unlabeled
---|---|---|---|---|---|---|---
politics|FGPS|766|5|sentence|Political propositions|not given|142,654
politics|POLT|318,761|2|paragraph|Political tweets|not given|none

## requirements

See `../../requirement.txt`

## file structure
- `setup.sh`: shell(bash) script to download original data files, convert them to json format, generate their shared vocabulary, and generate TFRecord files according to that vocabulary
- `write_tfrecord_merged.py`: python script to generate merged vocabulary and write TFRecord data files
- `args_merged`: arguments for the datasets that use the shared vocabulary
- `write_tfrecord_single.py`: python script to generate TFRecord files for the single dataset(without share vocabulary)
- `args_FGPS`: arguments for the dataset FGPS
- `args_POLT`: arguments for the dataset POLT
- `data/raw/`: downloaded/unzipped original data files
- `data/json/`: converted json data and basic vocabulary of the dataset
- `data/tf/merged/min_(min_freq)_max_(max_freq)`: generated data for the given min/max vocab frequency
    - `vocab_freq.json`: frequency of all the words that appeared in the training data(merged vocabulary)
    - `vocab_v2i.json`: mapping from word to id of the used vocabulary(only words appeared > min_frequency and < max_frequency)
    - `vocab_i2v.json`: mapping from id to word(sorted by frequency) of the used vocabulary
    - `dataset_name`
        - `train.tf`, `valid.tf`, `test.tf`: train/valid/test TFRecord files
        - `unlabeled.tf`: unlabeled TFRecord files(if there is unlabeled data)
        - `args.json`: arguments of the dataset
- `data/tf/single/dataset_name/min_(min_freq)_max_(max_freq)`: generated data for the given min/max vocab frequency for the single dataset
    - `train.tf`, `valid.tf`, `test.tf`: train/valid/test TFRecord files
    - `unlabeled.tf`: unlabeled TFRecord files(if there is unlabeled data)
    - `args.json`: arguments of the dataset
    - `vocab_freq.json`: frequency of all the words that appeared in the training data
    - `vocab_v2i.json`: mapping from word to id of the used vocabulary(only words appeared > min_frequency and < max_frequency)
    - `vocab_i2v.json`: mapping from id to word(sorted by frequency) of the used vocabulary


## steps

1. modify arguments in `args_merged.json`
2. run `setup.sh` to generate TFRecord files with merged vocabulary
3. (optional)
    - modify arguments in `args_FGPS.json` or `args_POLT.json`
    - generate TFRecord files for the single datasetrun by
        - `python ../write_tfrecords_single.py FGPS`
        - `python ../write_tfrecords_single.py POLT`

### arguments for datasets
- `max_document_length`: maximum document length for the word id(-1 if not given, only useful when padding is true)
- `min_frequency`: minimum frequency to keep when generating the vocabulary
- `max_frequency`: maximum frequency to keep when generating the vocabulary
- `label_field_name`: name of the label column in the json data file
- `text_field_names`: (list)names of the text columns in the json data file
- `train_ratio`: how much data to use as training data if no splits given
- `valid_ratio`: how much data out of all the data to use as validation data if no splits are given, or how much data out of training data to use as validation data if only train/test splits are given
- `random_seed`: seed used in random spliting indices
- `subsample_ratio`: how much data to use out of all the data
- `padding`: whether to pad the word ids
- `write_bow`: whether to write bag of words in the TFRecord file



## MLP baseline
type|dataset|accuracy|min_freq
---|---|---|---
political|FGPS|?|1
political|POLT|?|50

### hyperparameters:
- ?

### state-of-the-art results

- FGPS: ?

- POLT: ?

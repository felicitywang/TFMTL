## Overview

Information about the experiments.

## Datasets

type|name|#items|#labels|unit|summary|split|unlabeled
---|---|---|---|---|---|---|---
sentiment|SSTb|11,855|5|sentence|Rotten Tomatoes movie reviews|train:valid:test=8544:1101:2210|none
sentiment|LMRD|50,000|2|document|IMDB movie reviews|train:test=25,000:25,000|50,000
sentiment|IMDb|600,000|2|paragraph|IMDb movie reviews|train:test=300,000:300,000|none
sentiment|RTU|739,903|2|paragraph|Rotten Tomatoes user movie reviews|train:test=737,903:2000|none
sentiment|RTC|43,800|2|sentence|Rotten Tomatoes critic movie reviews|train:test=43,600,2000|none
sentiment|SUBJ|10,000|2|sentence|Rotten Tomatoes and IMDB movie reviews|not given|none
politics|FGPS|766|5|sentence|Political propositions|not given|142,654
politics|POLT|318,761|2|paragraph|Political tweets|not given|none

## Requirements

See `../../requirement.txt`

## File structure
- `task/`: different dataset combinations, e.g., `sentiment_1`, `politics_1`
    - `setup.sh`: shell(bash) script to download original data files, convert them to json format, generate their shared vocabulary, and generate TFRecord files according to that vocabulary
    - `args_merged`: arguments for the datasets that use the shared vocabulary
    - `args_DATASET`: arguments for the single dataset, e.g. `args_SSTb.json`, `args_LMRD.json`, etc.
    - `data/`: data files used in the experiment
        - `data/raw/`: downloaded/unzipped original data files
        - `data/json/`: converted json data and basic vocabulary of the dataset
        - `data/tf/`: TFRecord files
                - `data/tf/merged/DATASETXXX_DATASETYYY/min_(min_freq)_max_(max_freq)/`: generated data for the given min/max vocab frequency, e.g. `data/tf/merged/LMRD_SSTb/min_50_max_-1/`
                    - `vocab_freq.json`: frequency of all the words that appeared in the training data(merged vocabulary)
                    - `vocab_v2i.json`: mapping from word to id of the used vocabulary(only words appeared > min_frequency and < max_frequency)
                    - `vocab_i2v.json`: mapping from id to word(sorted by frequency) of the used vocabulary
                    - `DATASET/`
                        - `train.tf`, `valid.tf`, `test.tf`: train/valid/test TFRecord files
                        - `unlabeled.tf`: unlabeled TFRecord file(if there is unlabeled data)
                        - `args.json`: arguments used to generate the TFRecord files
                - `data/tf/single/DATASET/min_(min_freq)_max_(max_freq)`: generated data for the given min/max vocab frequency for the single dataset, e.g., `data/tf/single/LMRD/min_50_max_-1/`
                    - `train.tf`, `valid.tf`, `test.tf`: train/valid/test TFRecord files
                    - `unlabeled.tf`: unlabeled TFRecord files(if there is unlabeled data)
                    - `args.json`: arguments used to generate the TFRecord files
                    - `vocab_freq.json`: frequency of all the words that appeared in the training data
                    - `vocab_v2i.json`: mapping from word to id of the used vocabulary(only words appeared > min_frequency and < max_frequency)
                    - `vocab_i2v.json`: mapping from id to word(sorted by frequency) of the used vocabulary
- `scripts/`: scripts to run the experiments
    - `write_tfrecord_merged.py`: python script to generate merged vocabulary and write TFRecord data files
    - `write_tfrecord_single.py`: python script to generate the TFRecord files for the single dataset(without share vocabulary)
    - `write_tfrecord_predict.py`: python script to generate the TFRecord file for the given json file of the unlabeled text to predict
    - `write_tfrecord_test.py`: python script to generate the TFRecord file for the given json file of the labeled text to test
    - `convert_TEXT_to_JSON.py`: python script to convert to text to predict from plain text to gzipped json
    - `discriminative_driver.py`: driver script to run the MULT model

- restore saved checkpoints and evaluate on test data(e.g., `sentiment_1/test_mult.sh`)
- (restore saved checekpoints and predict given text)(e.g., `sentiment_1/predict_mult.sh`)


## Steps

### 1. Setup data

1. download data
2. convert original data files into gzipped json format
3. write TFRecord files from the json file

- for multiple datasets
    1. modify arguments in `TASK/args_merged.json`
    2. run `TASK/setup.sh` to generate TFRecord files with merged vocabulary
- for single dataset
    1. modify arguments in `TASK/args_DATASET.json`, e.g. `sentiment_1/args_SSTb.json`
    2. use `scripts/write_tfrecords_single.py` to generate TFRecord files, e.g. in `sentiment_1/` run `python ../scripts/write_tfrecords_single.py SSTb`

- if errors like `UnicodeDecodeError: 'ascii' codec can't decode byte xxxx in position xxxx: ordinal not in range(128)` occur, try setting system variable `export LC_ALL='en_US.utf8'`

### 2. Train the model

- train the MULT model with single/multiple datasets with training data and validation data, save the checkpoints
- run `scripts/discriminative_driver.py` with `train` mode with different configurations, e.g., `sentiment_1/train_mult.py`, `sentiment_1/test_mult.py`; see the source file for further hyper-parameter / argument explanation
- checkpoints will be saved to use in the `test` and `predict` mode

### 3. Test the model

- use the trained model to evaluate on the test data
- run `scripts/discriminative_driver.py` with `test` mode, specifying path of the saved checkpoints, e.g., `sentiment_1/test_single.sh`, `sentiment_1/test_mult.sh`; see the source file for further hyper-parameter / argument explanation

### 4. Predict with the model

<!-- TODO other features? -->

- use the trained model to give predictions of the given text
- input: plain text file, each line is a piece of data to predict
- run `scripts/convert_TEXT_to_JSON.py`, convert plain text to json(gzip) file, e.g., in `sentiment_1/`, run `python ../scripts/convert_TEXT_to_JSON.py ../../tests/LMRD_neg.txt data/raw/LMRD_neg.json.gz`
- run `scripts/write_tfrecords_predict.py`, write TFRecord file for the text to predict

    - e.g., in each task, `python ../scripts/write_tfrecords_predict.py DATASET_NAME predict_json_path predict_tf_path tfrecord_dir`
    - e.g., in `sentiment_1/`, run

        - `python ../scripts/write_tfrecords_predict.py args_LMRD.json data/raw/LMRD_neg.json.gz data/raw/LMRD_neg_single.tf data/tf/single/LMRD/min_50_max_-1/ data/tf/single/LMRD/min_50_max_-1/`
        - `python ../scripts/write_tfrecords_predict.py args_merged.json data/raw/LMRD_neg.json.gz data/raw/LMRD_neg_mult.tf data/tf/merged/LMRD_SSTb/min_50_max_-1/LMRD/ data/tf/single/LMRD/min_50_max_-1/`
- run `scripts/discriminative_driver.py` with `predict` mode, specifying path of the saved checkpoints, e.g., `sentiment_1/predict_single.sh`, `sentiment_1/predict_mult.sh`; see the source file for further hyper-parameter / argument explanation

### 5. Test with the model

- use the trained model to test other test data(except for the original test data)
- input: json file of test data with texts and labels
- run `scripts/write_tfrecords_test.py` to write TFRecord file for the data to test
    - e.g. `python ../scripts/write_tfrecords_test.py test_json_dir tfrecord_dir vocab_dir` where `test_json_dir` is the directory with the test json.gz file, `tfrecord_dir` is the directory to put in the TFRecord file, and `vocab_dir` is the directory of the vocabulary used in the model you're going to use(e.g. in `data/tf/merged/xxx/`)
- test the model following instructions in step 3, changing dataset path to the path where you write the extra test data

## Arguments to generate TFRecord files

- should be modified before generating the TFRecord files
- e.g., `TASK/args_merged.json` or `TASK/args_DATASET.json`

- `max_document_length`: maximum document length for the word id(-1 if not given, only useful when padding is true)
- `max_vocab_size`: maximum size allowed for the generated vocabulary
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
- `write_tfidf`: whether to write tf-idf in the TFRecord file(super slow, not recommended)


## Arguments of the generated TFRecord files

- generated after writing TFRecord files
- e.g., in 'TASK/data/tf/merged/DATASETXXX_DATASETYYY/min_(min_freq)_max_(max_freq)/DATASETXXX'

- num_classes: number of class labels
- max_document_length: maximum document length used to generate the TFRecord files
- vocab_size: size of the vocabulary used to generated the TFRecord files
- min_frequency: minimum frequency to keep when generating the vocabulary
- max_frequency: maximum frequency to keep when generating the vocabulary
- random_seed: seed used to split indices
- train_path: path to the TFRecord file of the training data
- valid_path: path to the TFRecord file of the validation data
- test_path: path to the TFRecord file of the test data
- train_size: size of the training data
- valid_size: size of the validation data
- test_size: size of the test data
- has_unlabeled: whether there is unlabeled data
- unlabeled_path: path to the TFRecord file of the unlabeled data
- unlabeled_size: size of the unlabeled data


## Baseline

type|dataset|accuracy|min_freq|model
---|---|---|---|---
sentiment|SSTb|40.7240%|1|CNN
sentiment|LMRD|89.0160%|50|BoW MLP
<!-- other datasets with more encoders -->

<!-- ### hyperparameters:
- learning rate: 0.0001
- dropout rate:0.5
- batch size: 32
- seed: 42
- max_num_epoch: 20(with early stopping)
- layers: [100, 100]
- encoding: bag of words
- train:valid = 9:1 if no valid split given -->

### State-of-the-art results for each dataset

- SSTb:

Tree-LSTM: 50.1%
http://aihuang.org/static/papers/AAAI2018_ClassifyAndStructure.pdf

original: 45.7%
https://nlp.stanford.edu/sentiment/code.html

others:
https://github.com/magizbox/underthesea/wiki/DATA-SST

- LMRD:

NgramCNN: 91.2%
http://porto.polito.it/2695485/1/ErionCanoIcgda2018_CR.pdf

original: 88.89%
http://ai.stanford.edu/~amaas/data/sentiment/

<!-- other datasets -->
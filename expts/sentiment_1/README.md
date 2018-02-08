## datasets

type|name|#items|#labels|unit|summary|split
---|---|---|---|---|---|---
sentiment|SSTb|11,855|5|sentence|Rotten Tomatoes movie reviews|train:valid:test=8544:1101:2210
sentiment|LMRD|50,000|2|document|IMDB movie reviews|train:test=25,000:25,000

## requirements
- pathlib==1.0.1
- nltk==3.2.5
- pytreebank==0.2.4
- tqdm==4.19.5


## file structure
- `setup.sh`: shell(bash) script to download original data files, convert them to json format, generate their shared vocabulary, and generate TFRecord files according to that vocabulary
- `write_tfrecord_merged.py`: python script to generate merged vocabulary and write TFRecord data files
- `args_merged`: arguments for the datasets that use the shared vocabulary
- `write_tfrecord_single.py`: python script to generate TFRecord files for the single dataset(without share vocabulary)
- `args_SSTb`: arguments for the dataset SSTb
- `args_LMRD`: arguments for the dataset LMRD
- `data/raw/`: downloaded/unzipped original data files
- `data/json/`: converted json data and basic vocabulary of the dataset
- `data/tf/merged/min_(min_freq)_max_(max_freq)`: generated data for the given min/max vocab frequency
    - `vocab_freq.json`: frequency of all the words that appeared in the training data(merged vocabulary)
    - `vocab_v2i.json`: mapping from word to id of the used vocabulary(only words appeared > min_frequency and < max_frequency)
    - `vocab_i2v.json`: mapping from id to word(sorted by frequency) of the used vocabulary
    - `dataset_name`
        - `train.tf`, `valid.tf`, `test.tf`: train/valid/test TFRecord files
        - `args.json`: arguments of the dataset
- `data/tf/single/dataset_name/min_(min_freq)_max_(max_freq)`: generated data for the given min/max vocab frequency for the single dataset
    - `train.tf`, `valid.tf`, `test.tf`: train/valid/test TFRecord files
    - `args.json`: arguments of the dataset
    - `vocab_freq.json`: frequency of all the words that appeared in the training data
    - `vocab_v2i.json`: mapping from word to id of the used vocabulary(only words appeared > min_frequency and < max_frequency)
    - `vocab_i2v.json`: mapping from id to word(sorted by frequency) of the used vocabulary


## steps

1. modify arguments in `args_merged.json`
2. run `setup.sh` to generate TFRecord files with merged vocabulary
3. (optional)
    - modify arguments in `args_SSTb.json` or `args_LMRD.json`
    - run `python write_tfrecord.py SSTb` or `python write_tfrecord.py LMRD` to generate TFRecord files for the single dataset

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

## running time
- 17 minutes for `setup.sh` with `min_freq=50` and `write_bow=true`(given
args_merged.json) on my laptop(16G RAM, 500G SSD)
- 5 minutes for `setup.sh` with `min_freq=0` and `write_bow=true` on my laptop
- 18 minutes for `setup.sh` `with min_freq=50` and `write_bow=true`(grid b02)

## memory
- raw(original):
    - SSTb: 3.8M
    - LMRD: 308.3M(zipped: 84M)
- json:
    - SSTb: 1M
    - LMRD: 30M
- tf:
    - min_freq=0, write_bow=true
        - SSTb: 4.7G
        - LMRD: 20G
    - min_freq=50, write_bow=true
        - SSTb: 309M
        - LMRD: 1.3G


## MLP baseline
type|dataset|accuracy|min_freq
---|---|---|---
sentiment|SSTb|40.7240%|1
sentiment|LMRD|89.0160%|50

### hyperparameters:
- learning rate: 0.0001
- dropout rate:0.5
- batch size: 32
- seed: 42
- max_num_epoch: 20(with early stopping)
- layers: [100, 100]
- encoding: bag of words
- train:valid = 9:1 if no valid split given

### state-of-the-art results

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


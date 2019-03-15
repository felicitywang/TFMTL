# Overview

Detailed instructions on how to run an experiment.

# Requirements

For Python package requirements and other general info see `../../requirement.txt`.

# File structure
- `TASK/`: name of an experiment, e.g., `sentiment_1`, `all_EMNLP`, etc.
    - `setup.sh`: shell(bash) script to download original data files, convert them to json format, generate their shared vocabulary, and generate TFRecord files according to that vocabulary
    - `args_NAME.json`: arguments to write TFRecord data from json format, could be
        - `args_merged.json`: default file used when processing different datasets together and write the TFRecord data with their shared vocabulary, `../scripts/write_tfrecords_merged.py` will look for this file in default if no other filenames specified
        - `args_DATASET`: arguments for each dataset separately, e.g. `args_SSTb.json`, `args_LMRD.json`, etc., `../scripts/write_tfrecords_single.py` will look for this file in default if no other filenames specified
        - `args_ANY-NAME-YOU-WANT.json`: need to specify the name
    - `data/`: data files used in the experiment
        - `data/raw/`: downloaded/unzipped original data files, etc.
        - `data/json/`: converted json data and basic vocabulary files of the dataset
            - `data.json.gz`: json data that should include text field and label field(if labeled)
            - `index.json.gz`(optional): train/dev/test splits
        - `data/tf/`: TFRecord files
                - `data/tf/merged/DATASETA_DATASETB_.../min_(min_freq)_max_(max_freq)_vocab_(max_vocab_size)_doc_(max_doc_len)_tok_(tokenizer_name)/`: generated data using particular arguments, e.g. `data/tf/merged/LMRD_SSTb/min_0_max_-1_vocab_10000_doc_-1_tok_tweet/`
                    - `vocab_freq.json`: frequency of all the words that appeared in the training data(merged vocabulary)
                    - `vocab_v2i.json`: mapping from word to id of the used vocabulary(only words appeared > min_frequency and < max_frequency)
                    - `vocab_i2v.json`: mapping from id to word(sorted by frequency) of the used vocabulary
                    - `DATASET/`
                        - `train.tf`, `valid.tf`, `test.tf`: train/valid/test TFRecord files
                        - `unlabeled.tf`: unlabeled TFRecord file(if there is unlabeled data)
                        - `args.json`: arguments used to generate the TFRecord files
                - `data/tf/single/DATASET/min_(min_freq)_max_(max_freq)_doc_(max_doc_len)_tok_(tokenizer_name)`: generated data for the given min/max vocab frequency and limited maximum vocabulary size for the single dataset, e.g., `data/tf/single/LMRD/min_0_max_-1_vocab_10000_doc_-1_tok_tweet/`
                    - `train.tf`, `valid.tf`, `test.tf`: train/valid/test TFRecord files
                    - `unlabeled.tf`: unlabeled TFRecord files(if there is unlabeled data)
                    - `args.json`: arguments used to generate the TFRecord files
                    - `vocab_freq.json`: frequency of all the words that appeared in the training data
                    - `vocab_v2i.json`: mapping from word to id of the used vocabulary(only words appeared > min_frequency and < max_frequency)
                    - `vocab_i2v.json`: mapping from id to word(sorted by frequency) of the used vocabulary
- `scripts/`: scripts to run the experiments
    - `write_tfrecord_single.py`: python script to generate the TFRecord files for the single dataset(without shared vocabulary)
    - `write_tfrecord_merged.py`: python script to generate merged vocabulary and write TFRecord data files for more than one datasets
    - `write_tfrecord_predict.py`: python script to generate the TFRecord file for the given json file of the unlabeled text to predict
    - `write_tfrecord_test.py`: python script to generate the TFRecord file for the given json file of the labeled text to test
    - `write_tfrecord_finetune.py`: python script to generate the TFRecord file for the given json file of a dataset to fine-tune the model with based on the dataset the model was pre-trained on
    - `convert_TEXT_to_JSON.py`: python script to convert to text to predict from plain text to json format
    - `discriminative_driver.py`: driver script to run the MUTL model'



# Pipeline

All the following commands suppose that you're in some experiment folder `expts/TASK/` and uses relative paths.

All the example commands can be found in `expts/example/`. (Run `python get_encoders.py` to generate `encoders.json`.)

## 1. Setup data

Run `./setup.sh` to setup the data.

1. Download data file in its original format using some public link
    - When writing your own `setup.sh`, remember to use some public links so that the experiment can be easily replicated

2. Convert original data files into gzipped json format
    - For existing datasets in this repo, this step would be `python .../convert_DATASET_to_JSON.py`

3. Write TFRecord files from the json file using different argument files (See `expts/example/setup.sh`.)

- relevant source files:
    - `mtl/util/dataset.py`
    - `expts/scripts/write_tfrecords_single.py`
    - `expts/scripts/write_tfrecords_merged.py`
- example files:
    - `expts/example/setup.sh`
- for a single dataset:
    - Modify default args file `args_DATASET.json`(e.g. `args_SSTb.json`) or use another name (e.g. `args_oneinput_nopretrain.json`)
    - Write TFRecord data with `python ../scripts/write_tfrecords_single.py DATASET [args_....json]`, e.g. `python ../scripts/write_tfrecords_single.py SSTb` or `python ../scripts/write_tfrecords_single.py SSTb args_oneinput_nopretrain.json`
- for multiple datasets
    - Modify default args file `args_merged.json` or use another name (e.g. `args_oneinput_nopretrain.json`)
    - Write TFRecord data with `python ../scripts/write_tfrecords_merged.py DATASET_1 DATASET_2 ... [args_....json]`, e.g. `python ../scripts/write_tfrecords_merged.py SSTb LMRD` or `python ../scripts/write_tfrecords_merged.py SSTb LMRD args_oneinput_nopretrain.json`
- if errors like `UnicodeDecodeError: 'ascii' codec can't decode byte bbbb in position bbbb: ordinal not in range(128)` occur, try setting system variable `export LC_ALL='en_US.utf8'`


## 2. Run Discriminative MULT

Run the MULT model with the discriminative driver scripts. Currently there're four modes:
- train: train the model on training set(`train.tf`) and evaluate on dev set(`valid.tf`) for certain epochs, saving the latest/best model
- test: test the trained model on test set(`test.tf`)
- predict: use the trained model to predict some unlabeled data
- finetune: finetune the pre-trained model on another dataset

Relevant source files:

- `mtl/models/mult.py`
- `expts/scripts/discriminative_driver.py`

### 2.0 Arguments

#### Required Arguments

* **Metadata**
    * `mode`: Either `train`, `test`, `predict`, or `finetune`
* **Training Details**
    * `alphas`: A list of decimals that sums to approximately 1. Each index corresponds to a specific dataset
    * `class_sizes`: A list of integers that represent how many classes are in each task
* **Logging Details**
    * `checkpoint_dir`: Folder to save trained models in `train` mode to or restore them from in `test`, `predict` and `finetune` modes
    * `log_file`: The file to store the log files
* **Dataset Details**
    * `topics_path`: List of paths to the data.json files. 1 path per dataset
    * `topics_field_name`: (name of keys for topic/input) - it defaults to `seq1`
    * `datasets`: List of names for each dataset
    * `dataset_paths`: List of paths to the TF records. 1 path per dataset
    <!-- * `vocab_size_file`: Path to the file that contains size of vocabulary, created when generating TFRecords. (moved to args.json)-->
* **Encoder Details**
    * `architecture`: The key for the json object in the corresponding encoder_config_file to use for encoders
    * `encoder_config_file`: The json file that contains the configurations for the encoders
* **Model_hyperparams**
    * `shared_hidden_dims`: List of integers, or just 1 integer that is broadcast for each layer
    * `private_hidden_dims`: List of integers, or just 1 integer that is broadcast for each layer
    * `shared_mlp_layers`: List of integers, or just 1 integer that is broadcast for each layer
    * `private_mlp_layers`: List of integers, or just 1 integer that is broadcast for each layer
* **Prediction** - When predicting:
    * `predict_tfrecord`: File path of the tf record file path of the text to predict
    * `predict_dataset`: Path to data to predict/annotate
    * `predict_output_folder`: Folder to save predictions
* **Finetune** - When fine-tuning:
    * `checkpoint_dir_init`: Path to load the pre-trained model from.

### Optional Arguments

<!-- TODO -->
* `experiment_name`: A string for the name of the experiment.


### 2.1 Train the model


Train the MULT model with single/multiple datasets with training data(`train.tf`) and validation data(`valid.tf`), save the checkpoints for the latest the best models. The models will be used in modes `test`,`predict` and `finetune`.

- Run `scripts/discriminative_driver.py` with `train` mode with different configurations, e.g.,
    - train single one-sequence-input dataset: `expts/example/train_SSTb_nopretrain.sh`
    - train two one-sequence-input datasets together: `expts/example/train_LMRD_SSTb_nopretrain.sh`
    - train single two-sequence-input dataset:
    `expts/example/train_Topic2_nopretrain.sh`
    - train two two-sequence-input datasets together:
    `expts/example/train_Target_Topic2_nopretrain.sh`


### 2.2. Test the model

Use the trained model to evaluate on the test set.

- Run `scripts/discriminative_driver.py` with `test` mode, specifying path of the saved checkpoints, e.g.,
    - test single one-sequence-input dataset: `expts/example/test_SSTb_nopretest.sh`
    - test two one-sequence-input datasets together: `expts/example/test_LMRD_SSTb_nopretest.sh`
    - test single two-sequence-input dataset:
    `expts/example/test_Topic2_nopretest.sh`
    - test two two-sequence-input datasets together:
    `expts/example/test_Target_Topic2_nopretest.sh`

### 2.3. Predict with the model

Use the trained model to predict unlabeled data. When writing TFRecord data for the unlabeled data to predict, remember to use the same vocabulary as the training data.

- Relevant source code
    - `expts/scripts/write_tfrecords_predict.py`

- Make sure the data to predict is also in json format. Apart from the text field(s), every example should also have a field `id` to help you distinguish the predictions. An example script that does this is `expts/scripts/convert_TEXT_to_JSON.py`. Run `python convert_TEXT_to_JSON.py text_file_path json_file_path`. e.g., `python ../scripts/convert_TEXT_to_JSON.py data/pred/SSTb_neg.txt data/pred/SSTb_neg.json.gz`
- Use `expts/scripts/write_tfrecords_predict.py` to write TFRecord data for it. Run `python write_tfrecords_predict.py dataset_args_path predict_json_path predict_tf_path vocab_dir`, e.g., `python ../scripts/write_tfrecords_predict.py args_SSTb.json data/pred/SSTb_neg.json.gz data/pred/SSTb_neg.tf data/tf/single/SSTb/min_1_max_-1_vocab_-1_doc_-1_tok_tweet/`
- Run the driver script in `predict` mode to use the trained classifier to give predictions, e.g. run `./predict_SSTb_nopretrain.sh`.
- Note that in the commands' arguments in the predict mode, `--datasets DATASET` means you're using the DATASET part of the trained model(private layers + output layer + the parameters that perform the best on the DATASET's dev set), thus the class sizes of the dataset to predict should be the same as DATASET, and the real TFRecord dataset name and data you predict using the saved model are passed in with `--predict_dataset` and `--predict_tfrecord_path`
- Outputs would be in both tsv and json formats and saved to `--predict_output_folder` as `PREDICT_DATASET.tsv` and `PREDICT_DATASET.json`. For each example, the output contains its id, predicted label and confidence scores between 0 and 1(softmax values of the output layer) for each label. If the task is binary classification, then there would only be confidence scores for the positive label(label "1"). e.g. Check the outputs `data/pred/SSTb_neg.pred/SSTb.json` and `data/pred/SSTb_neg.pred/SSTb.tsv`.

### 2.4. Test with the model

Apart from evaluating the classifier using the test set of the previous dataset, you can also write TFRecord files using its vocabulary for some other data you want to test the model with.

- Relevant source code:
    - `expts/scripts/write_tfrecords_test.py`

- The input should always be in JSON format. The required fields are texts and labels.
- To write the TFRecord files with the vocabulary of the training data the model has been trained with, run `python write_tfrecords_test.py args_test_json_path test_json_dir tfrecord_dir vocab_dir`, e.g., `python ../scripts/write_tfrecords_test.py args_SSTb.json data/test/SSTb_neg/json/ data/test/SSTb_neg/tf/ data/tf/single/SSTb/min_1_max_-1_vocab_-1_doc_-1_tok_tweet/`
- Run the discriminative driver script in `test` mode they way in 2.2, changing `--dataset_path` to the path the extra test data is saved into. e.g., run `test_SSTb_neg_nopretrain.sh`
- Similar to the `predict` mode, when testing some extra dataset other than the datasets used to train the model(e.g. testing `SSTb_neg` with the model trained with `SSTb` and `LMRD`), the `--datasets DATASET` argument refers to which part of the trained model to use; the real data are passed with `--dataset_paths`(e.g. in this example, the arguments should be `--dataset SSTb --dataset_paths path_to_SSTb_neg_data`, meaning you're using the SSTb' part of the model to evaluate the new SSTb_neg test data)

### 2.5. Finetune the model

The `finetune` mode lets you use initialize with a trained model and fine-tune it with another dataset. Currently only supports the STL with all the parameters shared(embedding layer, extracting layer, MLPs, output layer, etc.). Thus the two datasets should have the same number of classes.

- Relevant ource code
    - `expts/scripts/write_tfrecords_finetune.py`
- Write TFRecord for the new dataset, using the same vocabulary of the previous one. `expts/scripts/write_tfrecord_finetune.py` does this. Usage: `python write_tfrecord_finetune.py dataset_name args_finetune_json_path finetune_json_dir vocab_dir`, e.g.
    - `python ../scripts/write_tfrecords_finetune.py SSTb_finetune args_SSTb.json data/finetune/SSTb_neg/json/ data/tf/single/SSTb/min_1_max_-1_vocab_-1_doc_-1_tok_tweet/`
    - This command would write TFRecord data for `SSTb_neg` using the vocabulary used to write TFRecord data for `SSTb`, and save the TFRecord data in `data/tf/SSTb_finetune`
- Fine-tune the model with the new data with the `finetune` mode. You need to specify `checkpoint_dir_init`, which is the path where the pre-trained model is saved; and `checkpiont_dir`, which is the path whrer the fine-tuned model would be saved. e.g.,
    - `finetune_SSTb_neg_init_SSTb.sh` shows how to use the model pre-trained with SSTb and fine-tune it with the SSTb_neg data;
    - `test_finetune_SSTb_neg_init_SSTb.sh` shows how to use the fine-tuned model to test SSTb data



## Using pre-trained word embeddings

Pre-trained word embeddings are commonly used in NLP tasks nowadays. Four popular pre-trained word embeddings are supported in this library. Other formats of pre-trained word embeddings can also be easily used. To use a pre-trained word embedding file, one needs to first merge its vocabulary with the vocabulary generated solely from the training set of the dataset; then specify corresponding embedder and file paths in the encoder configuration file.

- Relevant source code
    - `pretrained_word_embeddings/*`
    - `mtl/embedders/pretrained.py`

### Steps
1. Currently the supported pre-trained word embeddings are Glove, fasttext, word2vec, word2vec slim. See `pretrained_word_embeddings/README.md` for more information.
2. Download pre-trained word embedding files using `pretrained_word_embeddings/download...`
3. Write TFRecord data: specify `pretrained_file` and `expand_vocab` in the args file. e.g. See `args_oneinput_glove_expand.json`, `args_oneinput_glove_init.json`
4. Write encoder configuration file:
    - for `embed_fn`, use either `expand_pretrained` or `init_pretrained`
        - `expand_pretrained`: expand training vocab with pre-trained file's vocabulary, new embeddings can be either trainable or not
        - `init_pretrained`: initialize training vocab with pre-trained word embeddings, new embeddings always trainable
    - for `embed_kwargs`:
        - make sure `embed_dim` matches the dimension of the pre-trained embeddings' dimension
        - specify `pretrained_path`, the path where the pre-trained word embedding file is saved
        - `trainable`: whether to fine-tune the part of word embeddings loaded from the pre-trained file(for words that aren't in the pre-trained file's vocabulary, their embeddings would be randomly initialized and always fine-tuned)

    - e.g.
        - `expand_pretiraned`
            - Write TFRecord data with `python ../scripts/write_tfrecords_single.py SSTb args_oneinput_glove_expand.json`
            - Train with `train_SSTb_glove_expand.py`
        - `init_pretiraned`
            - Write TFRecord data with `python ../scripts/write_tfrecords_single.py SSTb args_oneinput_glove_init.json`
            - Train with `train_SSTb_glove_init.py`

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
- e.g., in 'TASK/data/tf/merged/DATASETAAA_DATASETBBB/min_(min_freq)_max_(max_freq)/DATASETAAA'

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
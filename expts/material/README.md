# MATERIAL

Data descriptions and experiment pipeline for MATERIAL - DomainID Classification.

# About `mtl`

General introduction: `README.md`

Generation experiment introduction and pipeline: `expts/README.md`

Example files for `expts/README.md`: `expts/example/`

# About CLSP Grid

Make sure you've read these if you're using the grid to qsub
- General: http://wiki.clsp.jhu.edu/view/Introduction_to_the_CLSP_Grid
- Qsub: http://wiki.clsp.jhu.edu/view/A_gentle_introduction_to_sge_and_qsub
- GPU: http://wiki.clsp.jhu.edu/view/GPUs_on_the_grid

# Domains

| Domain | Description                | Gold Data |
| ------ | -------------------------- | --------- |
| GOV    | Government and Politics    | 1A, 1B    |
| LIF    | Lifestyle                  | 1A, 1B    |
| BUS    | Business and Commerce      | 1A        |
| LAW    | Law and Order              | 1A        |
| HEA    | Physical and Mental Health | 1B        |
| MIL    | Military                   | 1B        |
| SPO    | Sports                     | /         |


1A: Swahili

1B: Tagalog

# File Structure

- data/
    - raw/
        - TURK/
            - {BUS, GOV, HEA, LAW, LIF, MIL, SPO}.tsv: Turk sentences in tsv formats with averaged annotation scores in [0, 100]
        - gold/
            - labels/
                - DOMAIN/1{A,B}/domain_DOMAIN.list: list of filenames that are in-domain
            - translations/
                - {oracle, one, bop}/1{A,B}/{speech, text}/: human translation / one-best machine translation / bag-of-phrase machine translation
                    - filename.txt: document content
                    - filename.label: domains the document is in
    - json/
        - **DOMAIN_description**: data of domain DOMAIN, make sure your own datasets' names follow this format
        - **DOMAIN_gold_oracle**: gold data, oracle translation
        - **DOMAIN_gold_one**: gold data, one-best translation
        - **DOMAIN_gold_one**: gold data, bag-of-phrase translation
        - **DOMAIN_syn_p1000r1000**: Synthetic data, positive 1,000 examples, negative 1,000 examples
        - **DOMAIN_syn_p11000r11000**: Synthetic data, positive 11,000 examples, negative 11,000 examples
        - **DOMAIN_turk_90_50**: Turk data, filtered by positive cut 90 and negative cut 50
        - **DOMAIN_turk_80_50**: Turk data, filtered by positive cut 80 and negative cut 50
        - **DOMAIN_turk_70_50**: Turk data, filtered by positive cut 70 and negative cut 50
        - **DOMAIN_turk_60_50**: Turk data, filtered by positive cut 60 and negative cut 50
        - **DOMAIN_turk_50_50**: Turk data, filtered by positive cut 50 and negative cut 50
        - **DOMAIN_train_syn_p1000r1000_dev_gold_oracle**: combined data, train split syn_p1000r1000, dev split gold_oracle, (no test split)
        - **DOMAIN_train_syn_p11000r11000_dev_gold_oracle**: combined data, train split syn_p11000r11000, dev split gold_oracle, (no test split)
        - ...
        <!-- - TODO -->
    - tf/
        - single/
            - dataset_name/: as in data/json
                - min_(min_freq)_max(max_freq)_doc_(max_doc_len)_tok_(tokenizer_name)/
                    - {train, valid}.tf, etc.
    - pred/
        - json/
            <!-- - {doc,sent}/1{A,B}/{DEV, EVAL{1,2,3}, goldDOMAIN, ANALYSIS{1,2,3}}/{t6/mt-4.asr-s5,tt18, t6.bop/concat, tt18.bop/concat} -->
            - {doc,sent}/1{A,B}/{DEV, EVAL{1,2,3}, goldDOMAIN, ANALYSIS{1,2,3}}/.../
                - data.json.gz
        - tf/
            - {doc,sent}/1{A,B}/{DEV, EVAL{1,2,3}, goldDOMAIN, ANALYSIS{1,2,3}}/.../
                - dataset_name/: as in data/json/)
                    - min_(min_freq)_max(max_freq)_doc_(max_doc_len)_tok_(tokenizer_name)/
                        - pred.tf
    - results/
        - experiment_name/ (Optional)
            - dataset_name_- min_(min_freq)_max(max_freq)_doc_(max_doc_len)_tok_(tokenizer_name)_architecturename/
                - e: error log
                - o: output log
                - {train, test, predict, finetune}.sh: running scripts to submit
                - {train, test, predict, finetune}.log: log files
                - encoders.json: generated encoder file
                - ckpt/: folder to save the model
    - submissions/
        - subid_description/
            - 1A.txt
            - 1B.txt
            - DEV/{one,bop}/{1A,1B}/
                - d-domain.tgz
                - d-DOMAIN_NAME.tsv



# Pipeline

There're scripts that read in some configuration files in JSON format and generate running scripts/submit jobs for you. All you need to do is to be modify the configuration file. C-style comments (`//` at the beginning of a line or `/* */` surrounding a line) are allowed in these configurations files for your convenience(dependency: Python package `json_minify`).


## 0. Get json data

Run `setup_{gold, turk, syn}.py` to copy original files and convert them to json format to the current folder. Data would be saved in `data/json`.

`convert_{GOLD,TURK}_to_JSON.py` are examples that convert datasets from their original formats to JSON.


`combine_data_splits.py` can take two json datasets and combine them together as train and validation splits of one dataset.  Run `python combine_data_splits.py --train training_split_suffix --valid validation_data_split`, e.g. `python combine_data_spltis.py --train syn_p1000r1000 --valid gold_one` would use `data/json/DOMAIN_syn_p1000r1000/data.json.gz` as training split and `data/json/DOMAIN_gold_one/data.json.gz` as validation split for the six domains(except SPO, which doesn't have gold data.


## 1. Write TFRecord data

## 1.1 Write training data

Write TFRecord data for the data to train. Default split ratio is train : dev  = 9 : 1.

- Modify `args_nopretrain.json` / `args_pretrained.json` (or other args files)
    - Note: config files are in json files but allow C-style comments(`//` at the beginning of a line or `/* */` surrounding a line). It requires Python package `json_minify` to parse.
- Modify `write_train.json` (or use other files)(See comments).
    - Use your own environment and paths
    - dataset_name = domain + dataset_suffix
    - Indicate which args file you want to use
    - Note that in JSON file no extra `,` is allowed at the end(unlike in Python)
- Run `python get_write_train.py write_train.json` to get all the commands
    - Run all at once: `python get_write_train.py write_train.json > some_name.sh; bash some_name.sh`
    - Qsub and run in parallel: `bash split_qsub.sh get_write_train.py write_train.json some_tmp_name_prefix`, this will split the generated some_tmp_name_prefix.sh into some_tmp_name_prefixaa etc. and qsub them all to run in parallel.

## 1.2 Write data to predict

Write TFRecord data for the evaluation splits to predict according to the training data.

- Modify `args_file_path`(`args_nopretrain.json` / `args_nopretrained.json` or other such files). Only `text_field_names` and `label_field_names` will be used. Other arguments are read from the training data's `args.json`.
- Modify `write_pred.json` (See comments)
    - Same as 1.1
    - Choose which text_type, evaluation set, type of translation you want to predict
    - `args_path` is the `min_(min_freq)_max(max_freq)_doc_(max_doc_len)_tok_(tokenizer_name)` path generated automatically when writing the training data. This is how you decide which training data's vocabulary you want to use. Note that when using one model you should always make sure that you're using the same vocabulary.
- Run `python get_write_pred.py write_pred.json` to get all the commands
    - Run all at once: `python get_write_pred.py write_pred.json > some_name.sh; bash some_name.sh`
    - Qsub and run in parallel: `bash split_qsub.sh get_write_pred.py write_pred.json some_name`

## 1.3. Write data to fine-tune with

TODO

## 2. Run experiments

Basic pipeline:
- write TFRecord data with train and dev splits
- train the classifier
- maybe finetune the classifier with other data
- give predictions for the evaluation sets with the trained classifier
- submit the formatted predictions to the server
- collect results from the server to analyze

`qsub_stl_jobs.py` reads driver config file (e.g. `qsub_config_expt.json`) and encoder config file(e.g. `qsub_config_encod.json` generated by `python get_qsub_encod.py qsub_config_encod.json`) to generate encoder files and running scripts and submit jobs on the grid.


Make sure you're familiar with the grid when you use this, especially when using GPU.

## 2.0 Encoder Config File

`python get_qsub_encod.py qsub_config_encod.json` can generate a encoder configuration file that will be used generate encoder files for each dataset with a bunch of different encoders. You can directly modify `get_qsub_encod.py` following its current content to add more encoders.

## 2.1 Run the model

Refer to `expts/README.md` and examples in `expts/example/` for how to train/test a model.

Modify `qsub_config_expt.json`. See comments to see what each item does and which ones to modify.

Run `python qsub_stl_jobs.py qsub_config_expt.json qsub_config_encod.json` to write all the running scripts and maybe submit the jobs.

(TODO fine-tune mode to be added)

## 3. Get submissions

`get_server_submissions.py` will read the predictions and convert them to the format to submit to the server.

Modify `text_type` and `translation` in `get_server_submissions.py`. The default are `doc` and `one`.

`python get_server_submissions.py submission_config_folder eval_set` to read `1A.txt` and `1B.txt` in `submission_config_folder` and which evaluation sets to predict. `1A.txt` and `1B.txt` should include partial paths to find the prediction for positive threshold cut(above how much softmax to be considered in-domain) for each domain. e.g. one line should be like
```
GOV GOV_turk_60_50_min_1_max_-1_vocab_-1_doc_-1_tok_tweet_dan_meanmax_relu_0.1_nopretrain 0.5
```

This means that for the domain GOV, the predictions we want to use to submit are in `.../GOV_turk_60_50_min_1_max_-1_vocab_-1_doc_-1_tok_tweet_dan_meanmax_relu_0.1_nopretrain/...`(see File Structure for more details) and any files with confidence scores more than 0.5 would be kept as in-domain. The name is quite descriptive: the predictions are given by the model trained with GOV_turk_60_50 data preprocessed this way using architecture dan_meanmax_relu_0.1 without using pretrained word embeddings.

I recommend putting all such files in `submissions/subid_description/`, where the subid can be written into each submission's notes and help distinguish the results. e.g. See `submissions/20_turk_60_50_drop0.1_cut_0.5/1{A,B}.txt`.

The formatted files to submit would be saved in
submission_config_folder/eval_dir/translation/1{A,B}/d-domain.tgz

    - submissions/
        - subid_description/
            - 1A.txt
            - 1B.txt
            - DEV/{one,bop}/{1A,1B}/
                - d-domain.tgz
                - d-DOMAIN_NAME.tsv



## (Optional) 4. Parse server results

`parse_server_results.py` can convert the tsv files from the server to two lines of results in the order `BUS GOV-A GOV-B HEA LAW LIF-A LIF-B MIL SPO-A SPO-B` and delimitered with `,`, which can be directly copied and quickly pasted in the current Google sheets(paste it into the BUS column and `split text by columns`).

I recommend putting the two tsv files from the server into your `submission_folder`. e.g. `python parse_server_results.py submissions/20_turk_60_50_drop0.1_cut_0.5/` would give you this result:

```
BUS GOV-A GOV-B HEA LAW LIF-A LIF-B MIL SPO-A SPO-B
p_miss
0.0000%,27.2727%,54.0107%,38.5093%,21.7143%,0.0000%,0.6803%,13.2530%,9.0909%,21.9858%,
p_fa
100.0000%,34.8894%,17.9431%,47.8261%,48.4487%,100.0000%,100.0000%,67.5732%,3.7694%,7.9523%,
```



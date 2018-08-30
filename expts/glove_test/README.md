# Glove Usage

https://gitlab.hltcoe.jhu.edu/vandurme/tfmtl/issues/74

Usage and example of Glove pretrained word embeddings in tfmtl.

Branch: `pretrain`

Expts folder: `expts/glove_test/` or `expts/all_EMNLP/`

Steps:
1. download the pretrained word embeddings(Glove 6B and Glove Twitter 27B) with `pretrained_word_embeddings/`
2. write TFRecord files, specify `pretrained_file`(path) and `expand`(true if expand, otherwise init). Files would be saved to `data/tf/.../glove_name_{init, expand}/` e.g.`expts/glove_test/args_Topic2_glove.json`. Args file path can be passed into `write_tfrecords_xx.py` as the last argument.
3. write encoder file
    - set `embed_fn` to `init_glove` or `expand_glove`(code in `mtl/embedders/pretrained.py`)
    - set `trainable` of `embed_kwargs` (whether to fine-tune)
    - for `glove_init` set `reverse_vocab_path` to `data/tf/.../vocab_i2v.json`, set `random_size_path`
 to `data/tf/.../random_size.txt`(these are used to separate vocab in and not in Glove and to use `trainable`)
    - e.g. `expts/glove_test/encoders.json`
4. use corresponding dataset paths(with `init/expand` in the training scripts)
    - e.g. `expts/glove.test/train_Topic2_glove_{init,expand}.sh`
5. other examples would be `encoder_expts/encoders*.json` and `jobs/DATASET/train*.py` in `all_EMNLP/` created with `get_encoders.py` and `get_jobs.py`

# Datasets

- one input sequence: `Topic2`, `Target`
- two input sequences: `SSTb`, `LMRD`
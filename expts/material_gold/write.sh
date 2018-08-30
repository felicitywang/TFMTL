#!/usr/bin/env sh
# no pretrain
python ../scripts/write_tfrecords_single.py GOV_1000 args_nopretrain.json
python ../scripts/write_tfrecords_single.py LIF_1000 args_nopretrain.json
python ../scripts/write_tfrecords_single.py BUS_1000 args_nopretrain.json
python ../scripts/write_tfrecords_single.py LAW_1000 args_nopretrain.json
python ../scripts/write_tfrecords_single.py HEA_1000 args_nopretrain.json
python ../scripts/write_tfrecords_single.py MIL_1000 args_nopretrain.json
python ../scripts/write_tfrecords_single.py SPO_1000 args_nopretrain.json


# glvoe 6B 300d
python ../scripts/write_tfrecords_single.py GOV_1000 args_glove_expand.json
python ../scripts/write_tfrecords_single.py LIF_1000 args_glove_expand.json
python ../scripts/write_tfrecords_single.py BUS_1000 args_glove_expand.json
python ../scripts/write_tfrecords_single.py LAW_1000 args_glove_expand.json
python ../scripts/write_tfrecords_single.py HEA_1000 args_glove_expand.json
python ../scripts/write_tfrecords_single.py MIL_1000 args_glove_expand.json
python ../scripts/write_tfrecords_single.py SPO_1000 args_glove_expand.json


# fasttext 1M 300d
python ../scripts/write_tfrecords_single.py GOV_1000 args_fasttext_expand.json
python ../scripts/write_tfrecords_single.py LIF_1000 args_fasttext_expand.json
python ../scripts/write_tfrecords_single.py BUS_1000 args_fasttext_expand.json
python ../scripts/write_tfrecords_single.py LAW_1000 args_fasttext_expand.json
python ../scripts/write_tfrecords_single.py HEA_1000 args_fasttext_expand.json
python ../scripts/write_tfrecords_single.py MIL_1000 args_fasttext_expand.json
python ../scripts/write_tfrecords_single.py SPO_1000 args_fasttext_expand.json

# word2vec slim 300d
python ../scripts/write_tfrecords_single.py GOV_1000 args_word2vec_slim_expand.json
python ../scripts/write_tfrecords_single.py LIF_1000 args_word2vec_slim_expand.json
python ../scripts/write_tfrecords_single.py BUS_1000 args_word2vec_slim_expand.json
python ../scripts/write_tfrecords_single.py LAW_1000 args_word2vec_slim_expand.json
python ../scripts/write_tfrecords_single.py HEA_1000 args_word2vec_slim_expand.json
python ../scripts/write_tfrecords_single.py MIL_1000 args_word2vec_slim_expand.json
python ../scripts/write_tfrecords_single.py SPO_1000 args_word2vec_slim_expand.json

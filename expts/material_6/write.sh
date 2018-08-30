# no pretrain
python ../scripts/write_tfrecords_merged.py GOV LIF BUS HEA LAW MIL args_nopretrain.json

# glvoe 6B 300d
python ../scripts/write_tfrecords_merged.py GOV LIF BUS HEA LAW MIL args_expand_glove.json

# fasttext 1M 300d
# python ../scripts/write_tfrecords_merged.py GOV LIF BUS HEA LAW MIL args_expand_fasttext.json

# word2vec slim 300d
# python ../scripts/write_tfrecords_merged.py GOV LIF BUS HEA LAW MIL args_expand_word2vec_slim.json
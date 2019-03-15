python ../scripts/write_tfrecords_single.py SST2 args_train_1_oneinput.json
python ../scripts/write_tfrecords_single.py SST2 args_train_1_glove_expand_oneinput.json
python ../scripts/write_tfrecords_single.py SST2 args_train_1_glove_init_oneinput.json
python ../scripts/write_tfrecords_single.py SST2 args_train_1_glove_only_oneinput.json

python ../scripts/write_tfrecords_single.py RTC args_train_1_oneinput.json
python ../scripts/write_tfrecords_single.py RTC args_train_1_glove_expand_oneinput.json
python ../scripts/write_tfrecords_single.py RTC args_train_1_glove_init_oneinput.json
python ../scripts/write_tfrecords_single.py RTC args_train_1_glove_only_oneinput.json

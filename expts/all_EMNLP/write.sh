# min_frequency = 0 and all splits
python ../scripts/write_tfrecords_merged.py Stance Topic2 Topic5 ABSA-L ABSA-R MultiNLI FNC-1 Target args_merged_all_0.json

# min_frequency = 1 and only the training splits
python ../scripts/write_tfrecords_merged.py Stance Topic2 Topic5 ABSA-L ABSA-R MultiNLI FNC-1 Target args_merged_train_1.json

# python ../scripts/write_tfrecords_single.py Topic2 args_single.json
# python ../scripts/write_tfrecords_single.py Stance args_single.json



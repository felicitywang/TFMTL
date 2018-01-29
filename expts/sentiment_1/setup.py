from dataset import *

data_dirs = ["data/raw/SSTb/", "data/raw/IMDB/"]
merge_dict_write_tfrecord(data_dirs, new_data_dir="data/tf/")

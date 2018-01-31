from tasks.code.dataset import *

data_dirs = ["data/json/SSTb/", "data/json/LMRD/"]
merge_dict_write_tfrecord(data_dirs, new_data_dir="data/tf/")

from tasks.code.dataset import *

with open('args_merged.json', 'rt') as file:
    args_merged = json.load(file)
    file.close()

json_dirs = ["data/json/SSTb/", "data/json/LMRD/"]
tfrecord_dir = "data/tf/merged/"
tfrecord_dir += "min_" + str(args_merged['min_frequency']) + \
                "_max_" + str(args_merged['max_frequency'])
tfrecord_dirs = [tfrecord_dir + '/SSTb/', tfrecord_dir + '/LMRD/']

merge_dict_write_tfrecord(json_dirs=json_dirs,
                          tfrecord_dirs=tfrecord_dirs,
                          merged_dir=tfrecord_dir,
                          max_document_length=args_merged[
                              'max_document_length'],
                          min_frequency=args_merged['min_frequency'],
                          max_frequency=args_merged['max_frequency'],
                          train_ratio=args_merged['train_ratio'],
                          valid_ratio=args_merged['valid_ratio'],
                          subsample_ratio=args_merged['subsample_ratio'],
                          padding=args_merged['padding'],
                          write_bow=args_merged['write_bow'])

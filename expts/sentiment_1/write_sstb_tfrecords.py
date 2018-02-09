from tasks.code.dataset import *

data_dir = "data/json/SSTb/"
d = Dataset(data_dir, vocab_dir=None, tfrecord_dir=None, max_document_length=None,
        min_frequency=0, max_frequency=-1, write_bow=True)

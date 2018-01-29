## datasets

type|name|#items|#labels|unit|summary
---|---|---|---|---|---
sentiment|SSTb|11,855|5|sentence|Rotten Tomatoes movie reviews
sentiment|IMDB|50,000|2|document|IMDB movie reviews

## steps

1. download datasets

// TODO from original link

download the two datasets (data.json.gz and index.json.gz)from the grid and put into corresponding directories

/export/b02/fwang/mlvae/tasks/datasets/sentiment/SSTb/ -> ./data/raw/SSTb/

/export/b02/fwang/mlvae/tasks/datasets/sentiment/IMDB/ -> ./data/raw/IMDB/

2. run
`python3 setup.py`

(needs `dataset.py` from tasks/code/)
(run under this directory of hardcode path in `setup.py`)

## MLP baseline
type|dataset|accuracy|min_freq
---|---|---|---
sentiment|SSTb|40.7240%|1
sentiment|IMDB|89.0160%|50

### hyperparameters:
- learning rate: 0.0001
- dropout rate:0.5
- batch size: 32
- seed: 42
- max_num_epoch: 20(with early stopping)
- layers: [100, 100]
- encoding: bag of words


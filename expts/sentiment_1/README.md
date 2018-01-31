## datasets

type|name|#items|#labels|unit|summary|split
---|---|---|---|---|---|---
sentiment|SSTb|11,855|5|sentence|Rotten Tomatoes movie reviews|train:valid:test=8544:1101:2210
sentiment|LMRD|50,000|2|document|IMDB movie reviews|train:test=25,000:25,000

## steps

install python package `pytreebank`
run `setup.sh`

## MLP baseline
type|dataset|accuracy|min_freq
---|---|---|---
sentiment|SSTb|40.7240%|1
sentiment|LMRD|89.0160%|50

### hyperparameters:
- learning rate: 0.0001
- dropout rate:0.5
- batch size: 32
- seed: 42
- max_num_epoch: 20(with early stopping)
- layers: [100, 100]
- encoding: bag of words
- train:valid = 9:1 if no valid split given

### state-of-the-art results

- SSTb:

Tree-LSTM: 50.1%
http://aihuang.org/static/papers/AAAI2018_ClassifyAndStructure.pdf

original: 45.7%
https://nlp.stanford.edu/sentiment/code.html

others:
https://github.com/magizbox/underthesea/wiki/DATA-SST

- LMRD:

NgramCNN: 91.2%
http://porto.polito.it/2695485/1/ErionCanoIcgda2018_CR.pdf

original: 88.89%
http://ai.stanford.edu/~amaas/data/sentiment/


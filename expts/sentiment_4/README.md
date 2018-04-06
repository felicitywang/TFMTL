## Overview

See `../README.md` for file structure, code explanation and example commands.

## Datasets

type|name|#items|#labels|unit|summary|split|unlabeled
---|---|---|---|---|---|---|---
sentiment|SSTb|11,855|5|sentence|Rotten Tomatoes movie reviews|train:valid:test=8544:1101:2210|none
sentiment|RTC|43,800|2|sentence|Rotten Tomatoes critic movie reviews|train:test=43,600,2000|none
sentiment|RTU|739,903|2|paragraph|Rotten Tomatoes user movie reviews|train:test=737,903:2000|none

## Baseline

<!-- TODO -->
type|dataset|accuracy|min_freq
---|---|---|---
sentiment|SSTb|?|1
sentiment|IMDb|?|?
sentiment|RTU|?|?


### Hyper-parameters:

<!-- TODO -->
<!--
- learning rate: 0.0001
- dropout rate:0.5
- batch size: 32
- seed: 42
- max_num_epoch: 20(with early stopping)
- layers: [100, 100]
- encoding: bag of words
- train:valid = 9:1 if no valid split given -->

### State-of-the-art results

- SSTb:

Tree-LSTM: 50.1%
http://aihuang.org/static/papers/AAAI2018_ClassifyAndStructure.pdf

original: 45.7%
https://nlp.stanford.edu/sentiment/code.html

others:
https://github.com/magizbox/underthesea/wiki/DATA-SST

- IMDb:
?

- RTU:
?

## Overview

See `../README.md` for file structure, code explanation and example commands.

## Datasets

type|name|#items|#labels|unit|summary|split|unlabeled
---|---|---|---|---|---|---|---
sentiment|SSTb|11,855|5|sentence|Rotten Tomatoes movie reviews|train:valid:test=8544:1101:2210|none
sentiment|RTC|43,800|2|sentence|Rotten Tomatoes critic movie reviews|train:test=43,600,2000|none

## Baseline
type|dataset|accuracy|min_freq|model
---|---|---|---|---
sentiment|SSTb|40.7240%|1|CNN
<!-- TODO -->
<!-- sentiment|RTC|?|50|? -->

### Hyper-parameters
<!-- TODO -->


### State-of-the-art results

<!-- TODO -->

- SSTb:

Tree-LSTM: 50.1%
http://aihuang.org/static/papers/AAAI2018_ClassifyAndStructure.pdf

original: 45.7%
https://nlp.stanford.edu/sentiment/code.html

others:
https://github.com/magizbox/underthesea/wiki/DATA-SST

- RTC:
?

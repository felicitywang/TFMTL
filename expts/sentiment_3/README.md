## Overview

See `../README.md` for file structure, code explanation and example commands.

## Datasets

type|name|#items|#labels|unit|summary|split|unlabeled
---|---|---|---|---|---|---|---
sentiment|IMDb|600,000|2|paragraph|IMDb movie reviews|train:test=300,000:300,000|none
sentiment|RTU|739,903|2|paragraph|Rotten Tomatoes user movie reviews|train:test=737,903:2000|none

## Baseline

<!-- TODO -->

type|dataset|accuracy|min_freq|model
---|---|---|---|---
sentiment|IMDb|40.7240%|1|?
sentiment|RTU|?|100|?

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

<!-- TODO -->

- IMDb:
?

- RTU:
?

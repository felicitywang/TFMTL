## Overview

See `../README.md` for file structure, code explanation and example commands.

## Datasets

type|name|#items|#labels|unit|summary|split|unlabeled
---|---|---|---|---|---|---|---
politics|FGPS|766|5|sentence|Political propositions|not given|142,654
politics|POLT|318,761|2|paragraph|Political tweets|not given|none

## Baseline

<!-- TODO -->

type|dataset|accuracy|min_freq
---|---|---|---
political|FGPS|?|1
political|POLT|?|50

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

- FGPS: ?

- POLT: ?

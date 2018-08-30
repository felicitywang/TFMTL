# Overview

See `../README.md` for detailed explanation of the general experiments' setup and pipeline.

# Data

Make sure you have all the json data for SWts and LMRD in
`data/json/SWts/`
`data/json/LMRD/`

# Vocabulary arguments
Minimum and maximum frequency has little meaning here(now that we have max_vocab_size implemented) so they're set to default(0 and -1). Vocabulary size is limited to 10k for the document-level data(both SWts and LMRD).

# Train
`./train_single.sh` (train SWts solely)
`./train_mult.sh` (train SWts + LMRD)

# Test
`./test_single.sh` (test SWts)
`./test_mult.sh` (test SWts + LMRD)

# Predict

Follow previous instructions(`expts/README.md`) if you need to predict.

# Test extra

`./write_extra_single.sh` shows an example of writing TFRecord files for extra test data(currently it's just all the SWts data) with the vocabulary used for SWts single model. It assume that the TFRecord files are saved in `data/tf/extra_single/`

`./write_extra_mult.sh` shows ... with the vocabulary used for SWts + LMRD mult model. ... in `data/tf/extra_mult/`

`./test_extra.sh` has some examples evaluting different test data with different saved models. 
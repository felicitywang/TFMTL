# Pre-trained Word Embeddings

# Supported pretrained word embeddigns:
- Glove: https://nlp.stanford.edu/projects/glove/
- fasttext: https://fasttext.cc/docs/en/english-vectors.html
- word2vec: https://code.google.com/archive/p/word2vec/
- word2vec slim: https://github.com/eyaler/word2vec-slim

# File Structure

- `downlaod_{glove, word2vec, word2vec_slim, fasttext}.sh`: scripts to download and word embeddings files
- `{glove, word2vec, word2vec_slim, fasttext}/`: folders to put the downloaded files
- `../mtl/`
- `../expts/example/`: example training/testing scripts
- `../mtl/embedders/pretrained.py`: code to create embedding layer with the pre-trained word embeddings files
- `../mtl/util/load_embeds.py`: code to load the embedding files
- `../mtl/util/constants.py`: contains the list of all supported pretrained embedding file names (`VOCAB_NAMES`)
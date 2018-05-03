# Glove webpage with more download options:
# https://nlp.stanford.edu/projects/glove/


#!/bin/bash
file="../../pretrained_word_embeddings/glove/glove.6B.zip"
if [ -f "$file" ]
then
	echo "$file already exists."
else
    mkdir -p ../../pretrained_word_embeddings/glove/
    wget -dc http://nlp.stanford.edu/data/glove.6B.zip
    mv glove.6B.zip ../../pretrained_word_embeddings/glove/
fi

file="../../pretrained_word_embeddings/glove/glove.6B.50d.txt"
if [ -f "$file" ]
then
	echo "$file already exists."
else
    unzip ../../pretrained_word_embeddings/glove/glove.6B.zip -d ../../pretrained_word_embeddings/glove/
fi


# google
# wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

# google slim
# wget -c https://github.com/eyaler/word2vec-slim/raw/master/GoogleNews-vectors-negative300-SLIM.bin.gz
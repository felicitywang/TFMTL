# Download pretrained fasttext English models
# https://fasttext.cc/docs/en/english-vectors.html

#!/bin/bash

# wiki-news-300d-1M.vec.zip: 1 million word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens).

file="fasttext/wiki-news-300d-1M.vec.zip"
if [ -f "$file" ]
then
	echo "$file already exists."
else
    mkdir -p fasttext/
    wget -dc https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip
    mv wiki-news-300d-1M.vec.zip fasttext/
fi

# wiki-news-300d-1M-subword.vec.zip: 1 million word vectors trained with subword infomation on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens).

file="fasttext/wiki-news-300d-1M-subword.vec.zip"
if [ -f "$file" ]
then
	echo "$file already exists."
else
    mkdir -p fasttext/
    wget -dc https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M-subword.vec.zip
    mv wiki-news-300d-1M-subword.vec.zip fasttext/
fi


# crawl-300d-2M.vec.zip: 2 million word vectors trained on Common Crawl (600B tokens).

file="fasttext/crawl-300d-2M.vec.zip"
if [ -f "$file" ]
then
	echo "$file already exists."
else
    mkdir -p fasttext/
    wget -dc https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip
    mv crawl-300d-2M.vec.zip fasttext/
fi




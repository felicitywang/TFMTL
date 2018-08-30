# Glove webpage with more download options:
# https://nlp.stanford.edu/projects/glove/


#!/bin/bash

# Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip

file="glove/glove.6B.zip"
if [ -f "$file" ]
then
	echo "$file already exists."
else
    mkdir -p glove/
    wget -dc http://nlp.stanford.edu/data/glove.6B.zip
    mv glove.6B.zip glove/
fi

file="glove/glove.6B.100d.txt"
if [ -f "$file" ]
then
	echo "$file already exists."
else
    unzip glove/glove.6B.zip "glove.6B.100d.txt" -d glove/
fi


# Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download): glove.twitter.27B.zip
file="glove/glove.twitter.27B.zip"
if [ -f "$file" ]
then
	echo "$file already exists."
else
    mkdir -p glove/
    wget -dc http://nlp.stanford.edu/data/glove.twitter.27B.zip
    mv glove.twitter.27B.zip glove/
fi


file="glove/glove.twitter.27B.100d.txt"
if [ -f "$file" ]
then
	echo "$file already exists."
else
    unzip -j glove/glove.twitter.27B.zip "glove.twitter.27B.100d.txt" -d glove/
fi


# Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download): glove.42B.300d.zip

file="glove/glove.42B.300d.zip"
if [ -f "$file" ]
then
	echo "$file already exists."
else
    mkdir -p glove/
    wget -dc http://nlp.stanford.edu/data/glove.42B.300d.zip
    mv glove.42B.300d.zip glove/
fi

file="glove/glove.42B.300d.txt"
if [ -f "$file" ]
then
	echo "$file already exists."
else
    unzip -j glove/glove.42B.300d.zip "glove.42B.300d.txt" -d glove/
fi


# Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): glove.840B.300d.zip

file="glove/glove.840B.300d.zip"
if [ -f "$file" ]
then
	echo "$file already exists."
else
    mkdir -p glove/
    wget -dc http://nlp.stanford.edu/data/glove.840B.300d.zip
    mv glove.840B.300d.zip glove/
fi

file="glove/glove.42B.300d.txt"
if [ -f "$file" ]
then
	echo "$file already exists."
else
    unzip -j glove/glove.840B.300d.zip "glove.42B.300d.txt" -d glove/
fi

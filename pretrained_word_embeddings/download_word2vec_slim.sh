# Google word2vec slim version
# https://github.com/eyaler/word2vec-slim

file="word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz"
if [ -f "$file" ]
then
	echo "$file already exists."
else
    mkdir -p word2vec/
    wget -c https://github.com/eyaler/word2vec-slim/raw/master/GoogleNews-vectors-negative300-SLIM.bin.gz
    mv GoogleNews-vectors-negative300-SLIM.bin.gz word2vec/
fi

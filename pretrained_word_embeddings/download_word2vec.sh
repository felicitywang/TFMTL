# Google word2vec

file="word2vec/GoogleNews-vectors-negative300.bin.gz"
if [ -f "$file" ]
then
	echo "$file already exists."
else
    mkdir -p word2vec/
    wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    mv GoogleNews-vectors-negative300.bin.gz word2vec/

    # google word2vec vocab
    wget -c https://github.com/pvthuy/word2vec-GoogleNews-vocabulary/raw/master/vocabulary.zip
    mv vocabulary.zip word2vec/
    unzip word2vec/vocabulary.zip
fi



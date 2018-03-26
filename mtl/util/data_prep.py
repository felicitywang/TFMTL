# Copyright 2017 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Data Preprocessing"""

import itertools
import re
from collections import Counter

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer

# stemmer and tokenizer in NLTK
stemmer = PorterStemmer()


def stem_tokens(tokens, porter_stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(porter_stemmer.stem(item))
    return stemmed


tweet_tokenizer = TweetTokenizer(strip_handles=True,
                                 preserve_case=False,
                                 reduce_len=True)  # e.g. waaaayyyyyy -> waayyy


def tweet_clean(text):
    tokens = tweet_tokenizer.tokenize(text)
    return ' '.join(tokens)


def my_tokenizer(text):
    # remove punctuations other than ?!.
    # remove urls
    # text = re.sub(
    #     r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,'
    #     r'.;]+[-A-Za-z0-9+&@#/%=~_|][\r\n]*',
    #     ' ', text, flags=re.MULTILINE)
    # text=clean_str(text)
    tokens = tweet_tokenizer.tokenize(text)

    # stems = stem_tokens(tokens, stemmer)
    # return stems

    return tokens


# transform data['text'](string) to ngram model using
# sklearn.feature_extraction.text.TfidfVectorizer

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from:
        https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    not used
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def build_vocab(text_list):
    word_counts = Counter(itertools.chain(*text_list))
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    print(vocabulary)
    print(vocabulary_inv)
    return vocabulary, vocabulary_inv


def main():
    sentence = 'this is aaaaaaaaa a aaaaaa badly beautiful day . , / ? ! :) '
    print(tweet_clean(sentence))


if __name__ == "__main__":
    main()

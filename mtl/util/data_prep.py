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

from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import TweetTokenizer

"""STOP WORDS"""
NLTK_STOPWORDS = set(
  ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
   'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
   'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
   'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
   'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
   'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
   'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
   'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
   'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
   'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
   'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
   'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
   'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
   'will', 'just', 'don', 'should', 'now'])

"""Tokenizers"""
tweet_tokenizer = TweetTokenizer(strip_handles=True,
                                 preserve_case=False,
                                 reduce_len=True)  # e.g. waaaayyyyyy -> waayyy

tweet_tokenizer_keep_handles = TweetTokenizer(strip_handles=False,
                                              preserve_case=False,
                                              reduce_len=True)


def lower_tokenizer(tokens):
  return [i.lower() for i in tokens.split()]


def split_tokenizer(tokens):
  return tokens.split()


def ruder_tokenizer(xs, pattern="([\s'\-\.\,\!])", preserve_case=False):
  """Splits sentences into tokens by regex over punctuation: ( -.,!])["""
  tok = [x for x in re.split(pattern, xs)
         if not re.match("\s", x) and x != ""]
  if preserve_case:
    pass
  else:
    # THE COMMENTED OUT LINE BELOW IS NOT PYTHON 2 COMPATIBLE
    # tok = list(map(str.lower, tok))
    tok = [x.lower() for x in tok]
  return tok


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


"""Stemmers"""


def porter_stemmer(tokens):
  """Stem the tokens using nltk.stem.porter.PorterStemmer(
  https://www.nltk.org/api/nltk.stem.html), which follows the
  Porter stemming algorithm presented in
  Porter, M. “An algorithm for suffix stripping.” Program 14.3 (1980): 130-137.
  with some optional deviations that can be turned on or off with the mode
  argument to the constructor.

  :param tokens: a list of tokens
  :return: list of stemmed tokens
  """
  stemmer = PorterStemmer()
  stemmed = []
  for item in tokens:
    stemmed.append(stemmer.stem(item))
  return stemmed


def snowball_stemmer(tokens):
  """

  :param tokens:
  :return:
  """
  stemmer = EnglishStemmer()
  stemmed = []
  for item in tokens:
    stemmed.append(stemmer.stem(item))
  return stemmed


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
  string = re.sub(r"\\'", "\'", string)
  string = re.sub(r"\\""", "\"", string)
  string = re.sub(r"\s{2,}", " ", string)

  return string


def remove_urls(string):
  toks = []
  for tok in string.split():
    if re.match('https?:.*[\r\n]*', tok):
      tok = tok.split('http')[0]
      # keep everything before hand in cases where there is not space
      # between previous token and url
    if tok.strip():
      toks.append(tok)
  string = " ".join(toks)
  return string.strip()


def build_vocab(text_list):
  word_counts = Counter(itertools.chain(*text_list))
  vocabulary_inv = [word[0] for word in word_counts.most_common()]
  vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
  print(vocabulary)
  print(vocabulary_inv)
  return vocabulary, vocabulary_inv


def remove_tags(string):
  string = BeautifulSoup(string, "html5lib").get_text()
  return string


def remove_stopwords(tokens, stopwords, **kwargs):
  if stopwords == 'nltk':
    stopwords = NLTK_STOPWORDS
  else:
    raise NotImplementedError('No such stopwords as {}!'.format(stopwords))

  if 'weights' not in kwargs:
    tokens_kept = [token for token in tokens if token not in stopwords]
    # print('Removed {} stop words. (Before: {}; After: {}).'.format(len(
    #   tokens) - len(tokens_kept), len(tokens), len(tokens_kept)))
    return tokens_kept

  # TODO redundant ? (both of fixed max length)
  assert len(tokens) == len(kwargs['weights']), \
    'Token list(len {}) and weight list(len {}) are of different ' \
    'lengths!'.format(len(tokens), len(kwargs['weights']))

  tokens_kept = []
  weights_kept = []

  for token, weight in zip(tokens, kwargs['weights']):
    if token not in stopwords:
      tokens_kept.append(token)
      weights_kept.append(weight)

  # print('Removed {} stop words. (Before: {}; After: {}).'.format(len(
  #   tokens) - len(tokens_kept), len(tokens), len(tokens_kept)))
  return tokens_kept, weights_kept


def preproc(string):
  string = remove_urls(string)
  string = remove_tags(string)
  string = clean_str(string)
  return string


def main():
  sentence = 'this is aaaaaaaaa a aaaaaa badly beautiful day . , / ? ! \' :) ' \
             '" \' ' \
             '<img<!-- --> src=x onerror=alert(1);//><!-- -->' \
             '<ref/> test ref <ref>' \
             'http://www.test.com' \
             '<td><a href="http://www.fakewebsite.com">Please can you strip me?</a>' \
             '<br/><a href="http://www.fakewebsite.com">I am waiting....</a></td>'
  print(clean_str(sentence))
  print(tweet_clean(sentence))
  print(BeautifulSoup(sentence).get_text())
  print(preproc(sentence))


if __name__ == "__main__":
  main()

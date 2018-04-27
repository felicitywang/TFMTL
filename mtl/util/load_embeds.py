from __future__ import unicode_literals

import random

import numpy as np


def load_Glove(glove_path, train_vocab_list):
  """combine Glove vocabulary and training-data vocabulary, randomly

  initialize the latter
  Modified from
  https://gitlab.hltcoe.jhu.edu/research/tf-ner/blob/master/ner/embeds.py
  :param glove_path: path to the Glove file
  :param train_vocab_list: list, all the word types in the training data
  :return: word embedding numpy matrix and v2i mapping
  """
  print('Loading embeddings from {}...'.format(glove_path), end='')
  print('{} original vocabulary from training.'.format(len(train_vocab_list)))
  word2embed = {}
  file = open(glove_path, 'rb')
  found = 0
  last_dims = None
  for line_count, line in enumerate(file.readlines()):
    #        print(line)
    line = line.decode('utf-8').rstrip()
    # Google / w2v with tab separating word and vec
    # TODO compatible with other formats?
    if '\t' in line:
      splitted = line.split('\t')
      word = splitted[0]
      row = splitted[1].split(' ')
      dims = len(row)
    # GLoVe space-delim style file
    else:
      splitted = line.split(' ')
      word = splitted[0]
      row = splitted[1:]
      dims = len(row)
    # Convert to float and add to dict
    assert (last_dims is None or dims ==
            last_dims), "Dim mismatch parsing line {}:\n{}".format(
      line_count + 1, line)
    last_dims = dims
    if word in train_vocab_list:
      found += 1
    word2embed[word] = [float(f) for f in row]
  print("Embeddings found for {} words from train set.".format(found))

  embeds = []

  # Setup the UNK vector
  if '<unk>' in word2embed.keys():
    oov_vector = word2embed['<unk>']
  else:
    oov_vector = [random.uniform(-1.0, 1.0) for _ in range(dims)]

  # Setup the numpy embedding matrix
  for word in train_vocab_list:
    if word in word2embed:
      embeds.append(word2embed[word])
    else:
      embeds.append(oov_vector)

  remaining = list(set(word2embed.keys()) - set(train_vocab_list))
  for word in remaining:
    embeds.append(word2embed[word])
  file.close()
  print('Finished loading {} embeddings.'.format(len(embeds)))
  return np.asarray(embeds, dtype=np.float32), {w: i for i, w in enumerate(
    train_vocab_list + remaining)}


if __name__ == '__name__':
  import json

  vocab = json.load(open('vocab_v2i.json'))
  a, b = load_Glove('glove.6B.50d.txt', vocab)

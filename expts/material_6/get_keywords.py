import gzip
import json
import os

from tqdm import tqdm

from mtl.util.data_prep import tweet_tokenizer

"""Remove keywords from data.json"""

remove_words = {}

for filename in os.listdir('keywords/'):
  # print(filename)
  if len(filename) != 4:
    continue
  with open(os.path.join('keywords', filename)) as file:
    lines = file.readlines()
    assert len(lines) == 1, len(lines)
    line = lines[0].split('\t')
    dataset = line[0]
    keyphrases = line[1].strip().split('\"')
    keyphrases = [k.lower() for k in keyphrases if k != ' ' and k != '']
    # use separate words instead of exact phrase matches
    keywords = []
    for phrase in keyphrases:
      keywords.extend(phrase.split())
      keywords.extend(tweet_tokenizer.tokenize(' '.join(phrase)))
    remove_words[dataset] = keywords

all_words = set()
for dataset in remove_words:
  for word in remove_words[dataset]:
    all_words.add(word)
print('Old all words ', len(all_words))

not_founds = 0
founds = 0
for dataset in os.listdir('data/json/'):
  filename_bak = os.path.join('data/json/', dataset, 'data.bak.json.gz')
  filename = os.path.join('data/json/', dataset, 'data.json.gz')
  with gzip.open(filename_bak, 'rt') as fin:
    data = json.load(fin, encoding='utf-8')

    for index, item in tqdm(enumerate(data)):
      text = item['text']
      text = text.lower()
      found = False
      for word in remove_words[dataset]:
        if word in text:
          # print("Found ", word)
          found = True
          text = text.replace(word, ' ')
          if word in all_words:
            all_words.remove(word)
      if found:
        for word in remove_words[dataset]:
          if word in text:
            # print("Found ", word)
            print('Wrong!!!')
            text = text.replace(word, ' ')
            if word in all_words:
              all_words.remove(word)

      if not found:
        # print('Not found', text)
        not_founds += 1
      else:
        founds += 1
      item['index'] = index
      item['text'] = text
      # if len(text) != len(item['text']):
      #   print('Old:', len(text), 'New', len(item['text']))

  with gzip.open(filename, 'wt') as fout:
    json.dump(data, fout, ensure_ascii=False)

print(not_founds, founds)
print('Left all words ', len(all_words))
for all_word in all_words:
  print(all_word)

# check no keywords
not_founds = 0
founds = 0
for dataset in os.listdir('data/json/'):
  filename_bak = os.path.join('data/json/', dataset, 'data.json.gz')
  filename = os.path.join('data/json/', dataset, 'data.json.gz')
  with gzip.open(filename, 'rt') as fin:
    data = json.load(fin, encoding='utf-8')

    for index, item in tqdm(enumerate(data)):
      text = item['text']
      text = text.lower()
      found = False
      for word in remove_words[dataset]:
        if word in text:
          # print("Found ", word)
          found = True
          text = text.replace(word, ' ')
          if word in all_words:
            all_words.remove(word)
      if found:
        for word in remove_words[dataset]:
          if word in text:
            # print("Found ", word)
            print('Wrong!!!')
            text = text.replace(word, ' ')
            if word in all_words:
              all_words.remove(word)

      if not found:
        # print('Not found', text)
        not_founds += 1
      else:
        founds += 1
      item['index'] = index
      item['text'] = text
      # if len(text) != len(item['text']):
      #   print('Old:', len(text), 'New', len(item['text']))

  # with gzip.open(filename, 'wt') as fout:
  #   json.dump(data, fout, ensure_ascii=False)

assert founds == 0

print(not_founds, founds)
print('Left all words ', len(all_words))
# for all_word in all_words:
#   print(all_word)

"""Create gold sent data"""
import gzip
import json
import os

from mtl.util.util import make_dir


def main():
  gov_data = []
  lif_data = []
  gov_index = 0
  lif_index = 0
  dir = '/export/a05/mahsay/domain/goldstandard/sent'
  for lang in ['1A', '1B']:
    for domain in ['GOV', 'LIF']:
      for filename in os.listdir(os.path.join(dir, lang, domain)):
        with open(os.path.join(dir, lang, domain, filename)) as file:
          if domain == 'GOV':
            gov_data.append({
              'index': gov_index,
              'id': os.path.join(lang, domain, filename),
              'text': ' '.join(file.readlines()),
              'label': 1
            })
            gov_index += 1
          else:
            lif_data.append({
              'index': lif_index,
              'id': os.path.join(lang, domain, filename),
              'text': ' '.join(file.readlines()),
              'label': 1
            })
            lif_index += 1
  print(len(gov_data), len(lif_data))
  make_dir('data/json/gold_sent/GOV/')
  make_dir('data/json/gold_sent/LIF/')

  with gzip.open('data/json/gold_sent/GOV/data.json.gz', mode='wt') as file:
    json.dump(gov_data, file, ensure_ascii=False)

  with gzip.open('data/json/gold_sent/LIF/data.json.gz', mode='wt') as file:
    json.dump(lif_data, file, ensure_ascii=False)


if __name__ == '__main__':
  main()

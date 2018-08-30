"""Randomly take 200 of the 1000 negative examples"""
import gzip
import json
import os

import numpy as np
from tqdm import tqdm

random_seed = 42
np.random.seed(random_seed)

for dataset in os.listdir('data/json/'):
  filename = os.path.join('data/json/', dataset, 'data.json.gz')
  with gzip.open(filename, 'rt') as fin:
    data = json.load(fin, encoding='utf-8')

    pos_inds = []
    neg_inds = []

    for index, item in tqdm(enumerate(data)):
      text = item['text']
      label = int(item['label'])
      if label == 1:
        pos_inds.append(index)
      else:
        if len(text.split()) > 50:
          neg_inds.append(index)

    assert len(pos_inds) == 100, len(pos_inds)
    assert len(neg_inds) >= 100, len(neg_inds)

  # 100 neg examples
  neg_inds = np.split(neg_inds, [100])[0]
  new_data = [data[i] for i in pos_inds]
  new_data += [data[i] for i in neg_inds]
  assert len(new_data) == 200, len(new_data)

  with gzip.open(filename, 'wt') as fout:
    json.dump(new_data, fout, ensure_ascii=False)

for dataset in os.listdir('data/json/'):
  filename = os.path.join('data/json/', dataset, 'data.json.gz')
  with gzip.open(filename, 'rt') as fin:
    data = json.load(fin, encoding='utf-8')
    assert len(data) == 200, len(data)

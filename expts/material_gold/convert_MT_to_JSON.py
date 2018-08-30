"""Read the tsv MT data and convert to json format"""
import gzip
import json
import os
import sys

import pandas as pd

from mtl.util.util import make_dir


def main():
  assert len(sys.argv) == 4, 'Usage: python convert_MT_to_JSON.py din dout ' \
                             'threshold'

  din = sys.argv[1]
  dout = sys.argv[2]
  threshold = float(sys.argv[3])
  for filename in os.listdir(din):
    df = pd.read_csv(os.path.join(din, filename), sep='\t')
    data = []
    for index, item in enumerate(df.to_dict('records')):
      # print(index, item['mean_score'])
      data.append({
        'index': index,
        'id': item['id'],
        'text': item['sent'],
        'score': item['mean_score'] / 1000.0,
        'label': 0 if item['mean_score'] < threshold else 1
      })

    dir = os.path.join(dout, filename[:3] + '_' + str(threshold))
    make_dir(dir)
    with gzip.open(os.path.join(dir, 'data.json.gz'), mode='wt') as file:
      json.dump(data, file, ensure_ascii=False)


if __name__ == '__main__':
  main()

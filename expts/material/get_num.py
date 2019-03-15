import gzip
import json
import os
import sys

folder = sys.argv[1]
with gzip.open(os.path.join(folder, 'data.json.gz'), mode='rt') as file:
    data = json.load(file)
    for i in data:
        print(i['id'])
    print(len(data))

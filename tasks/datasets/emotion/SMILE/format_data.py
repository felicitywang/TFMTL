import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

file = open('smile-annotations-final.json')

data_list = []

for line in file.readlines():
    data_list.append(json.loads(line))

print(len(data_list))
with open('data.json', 'w') as file:
    json.dump(data_list, file)

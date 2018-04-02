import json
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

path = "./original/"
files = os.listdir(path)
count = 0
data_list = []
for emotion in files:
    file = open(path + emotion, "r")
    emotion = emotion.split("_")[2]
    emotion = emotion.split(".")[0]
    for line in file:
        data_list.append({'emotion': emotion, 'text': line.strip()})
        count += 1
json.dump(data_list, open('data.json', 'w'), ensure_ascii=False)
print(count)

df = pd.read_json('data.json')

label_encoder = LabelEncoder()
df['label'] = None
df.label = label_encoder.fit_transform(df.emotion)

print(label_encoder.classes_)
df.to_json('data.json', orient='records')

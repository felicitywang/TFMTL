import pandas as pd
from sklearn.preprocessing import LabelEncoder

file = open('Jan9-2012-tweets-clean.txt')

data_list = []

for line in file.readlines():
    line = line.split(':', 1)
    id = line[0]
    pos = line[1].rfind("::")
    text = line[1][:pos]
    emotion = line[1][pos:].split()[1]
    print(emotion)
    data_list.append({'id': id, 'text': text, 'emotion': emotion})


df = pd.DataFrame(data_list)
df['label'] = None
label_encoder = LabelEncoder()
df.label = label_encoder.fit_transform(df.emotion)
print(label_encoder.classes_)
print(len(data_list))
df.to_json('data.json', orient='index')


import os
import json
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

import os
import json
path = "./original/emotion/data/"
files = os.listdir(path)
count=0
data_list = []
for emotion in files:
    file = open(path + emotion, "r")
    for line in file:
        data_list.append({'emotion': emotion, 'text': line.strip()})
        count+=1
json.dump(data_list, open('data.json', 'w'))
print(count)

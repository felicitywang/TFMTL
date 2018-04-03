import json

file = open(
    'annotated-data-emotions/tweets_annotated_with_6_Ekman\'s_emotions')

data_list = []

for line in file.readlines():
    line = line.split('::', 1)
    text = line[0]
    emotion = line[1].split()[0]
    # print(text)
    # print(emotion)
    data_list.append({'text': text, 'emotion': emotion})

file = open('tweets_annotated_with_6_Ekman\'s_emotion_synonyms')
for line in file.readlines():
    pos = line.rfind('::')
    text = line[:pos]
    emotion = line[pos + 2:].split()[0]
    # print(text)
    # print(emotion)
    data_list.append({'text': text, 'emotion': emotion})

file = open('data.json', 'w')
json.dump(data_list, file)

# df = pd.DataFrame(data_list)
# df['label'] = None
# label_encoder = LabelEncoder()
# df.label = label_encoder.fit_transform(df.emotion)
# print(label_encoder.classes_)
# print(len(data_list))
# df.to_json('data.json', orient='index')

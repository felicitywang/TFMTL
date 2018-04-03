import gzip
import json

index = 0

train_list = json.load(open('train.json'))
for i in train_list:
    i['index'] = index
    index += 1

test_list = json.load(open('test.json'))
for i in test_list:
    i['index'] = index
    index += 1

all_list = train_list
all_list.extend(test_list)

with gzip.open('data.json.gz', mode='wt') as file:
    json.dump(all_list, file)

# indices
train_index = list(range(len(train_list)))
test_index = list(range(len(train_list), len(train_list) + len(test_list)))
index = {
    'train': train_index,
    'test': test_index
}
assert len(set(index['train']).intersection(index['test'])) == 0

with gzip.open('index.json.gz', mode='wt') as file:
    json.dump(index, file)

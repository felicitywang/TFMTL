# Copyright 2017 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

# runs under DailyDialogue-Parse-master/ folder

import json
import re
from os import listdir
import pandas as pd

text = open('dialogues_text.txt', 'r')
act = open('dialogues_act.txt', 'r')
emotion = open('dialogues_emotion.txt', 'r')
topic = open('dialogues_topic.txt', 'r')

all_list = []
index = 0
dialogue_id = 0
for text_line in text.readlines():
    texts = text_line.split('__eou__')[:-1]
    acts = act.readline().split()
    emotions = emotion.readline().split()
    topics = topic.readline().split()

    for i in range(len(texts) - 1):
        content = dict()
        content['dialogue_id'] = index
        content['index'] = index
        index += 1
        content['text'] = texts[i]
        content['act'] = acts[i]
        content['emotion'] = acts[i]
        content['topic'] = topics[0]
        content['label'] = content['act']
        all_list.append(content)
    dialogue_id += 1

with open('data.json', 'w') as file:
    json.dump(all_list, file)


text = open('train/dialogues_train.txt', 'r')
act = open('train/dialogues_act_train.txt', 'r')
emotion = open('train/dialogues_emotion_train.txt', 'r')

train_list = []
index = 0
for text_line in text.readlines():
    texts = text_line.split('__eou__')[:-1]
    acts = act.readline().split()
    emotions = emotion.readline().split()
    topics = topic.readline().split()

    for i in range(len(texts) - 1):
        content = dict()
        content['index'] = index
        index += 1
        content['text'] = texts[i]
        content['act'] = acts[i]
        content['emotion'] = acts[i]
        content['label'] = content['act']
        all_list.append(content)

with open('train.json', 'w') as file:
    json.dump(all_list, file)

validation_list = []
index = 0
for text_line in text.readlines():
    texts = text_line.split('__eou__')[:-1]
    acts = act.readline().split()
    emotions = emotion.readline().split()
    topics = topic.readline().split()

    for i in range(len(texts) - 1):
        content = dict()
        content['index'] = index
        index += 1
        content['text'] = texts[i]
        content['act'] = acts[i]
        content['emotion'] = acts[i]
        content['label'] = content['act']
        all_list.append(content)

with open('dev.json', 'w') as file:
    json.dump(all_list, file)


test_list = []
index = 0
for text_line in text.readlines():
    texts = text_line.split('__eou__')[:-1]
    acts = act.readline().split()
    emotions = emotion.readline().split()
    topics = topic.readline().split()

    for i in range(len(texts) - 1):
        content = dict()
        content['index'] = index
        index += 1
        content['text'] = texts[i]
        content['act'] = acts[i]
        content['emotion'] = acts[i]
        content['label'] = content['act']
        all_list.append(content)

with open('test.json', 'w') as file:
    json.dump(all_list, file)

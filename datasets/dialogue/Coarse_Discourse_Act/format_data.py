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

"""Reorganize data collected using Reddit API with join_forum_data.py

into python lists of dicts that stores different features.
Reddit body with annotations dump is stored in
    ../cache/coarse_discourse_dump_reddit.json

Add features to the original list of threads;
and create a new list of all the posts (in previous sequence) with more
features
Save the lists to ../cache/post_list.pickle and ../cache/thread_list.pickle
"""

import json
import pickle

import nltk

data_list = []

with open("../cache/coarse_discourse_dump_reddit.json") as data_file:
    for line in data_file.readlines():
        thread = json.loads(line)
        data_list.append(thread)
data_file.close()

# handle duplicates
num_duplicate_post = 0
duplicate_found = []
duplicate_lost = []

all_duplicate_id_set = set()

for thread in data_list:
    # find duplicate post in thread
    post_id_set = set()
    duplicate_id_set = set()
    for post in thread['posts']:
        if post['id'] in post_id_set:
            # print(post['id'])
            duplicate_id_set.add(post['id'])
            all_duplicate_id_set.add(post['id'])
        else:
            post_id_set.add(post['id'])
    if len(post_id_set) != len(thread['posts']):
        # print(thread['id'], "duplicate!")
        num_duplicate_post += len(thread['posts']) - len(post_id_set)

    # remove duplicate posts without body
    new_posts = []
    for post in thread['posts']:
        if post['id'] in duplicate_id_set and 'body' not in post:
            pass
        else:
            new_posts.append(post)
    thread['posts'] = new_posts

    # for post in thread['posts']:
    #     if post['id'] in duplicate_id_set:
    #         if 'body' not in post:
    #             thread['posts'].remove(post)

    # # check again
    # post_id_set = set()
    # duplicate_id_set = set()
    # for post in thread['posts']:
    #     if post['id'] in post_id_set:
    #         # print(post['id'])
    #         duplicate_id_set.add(post['id'])
    #     else:
    #         post_id_set.add(post['id'])
    # if len(duplicate_id_set) > 0:
    #     print(duplicate_id_set)
    #     print("error")
    #     exit()

    # duplicates found
    post_id = set()
    for post in thread['posts']:
        post_id.add(post['id'])

    for id in duplicate_id_set:
        if id in post_id:
            duplicate_found.append(id)
        else:
            duplicate_lost.append(id)

pickle.dump(all_duplicate_id_set, open("../cache/all_duplicate_id_set.pickle",
                                       "wb"))
pickle.dump(duplicate_found, open("../cache/duplicate_found.pickle", "wb"))

print("number of duplicates is", num_duplicate_post)
print("number of duplicates found is", len(duplicate_found))
print("number of duplicated lost is", len(duplicate_lost))

# for i in duplicate_lost:
#     print(i)

# thread_list
# thread(dict):
#     id(=id of initial post)
#     title
#     body
#     subreddit
#     is_self_post
#     number of all comments
#     list of all ids
#     avg of post depths
#     avg num of sentence per post
#     avg num of words per
#     avg num of

thread_list = []
post_list = []

for thread in data_list:

    # sum_thread_num_sent = 0
    # sum_thread_num_word = 0
    # sum_thread_num_char = 0
    sum_thread_post_depth = 0

    for post in thread['posts']:
        if 'is_first_post' in post:
            thread['id'] = post['id']
        else:
            post['is_first_post'] = False

    for post in thread['posts']:
        post['subreddit'] = thread['subreddit']
        post['thread_id'] = thread['id']
        if post['is_first_post']:
            post['title'] = thread['title']
        # post['text'] = ''
        # if 'title' in post:
        #     post['text'] += post['title']
        # if 'body' in post:
        #     post['text'] += post['body']

        # get number of sentences/words/characters of each post and each thread
        if 'post_depth' not in post:
            post['post_depth'] = 0
        # if 'text' in post:
        #     text = post['text']
            # sents = nltk.sent_tokenize(text)
            # tokens = nltk.word_tokenize(text)
            # post['num_sent'] = len(sents)
            # post['num_word'] = len(tokens)
            # post['num_char'] = len(text)

        # sum_thread_num_sent += len(sents)
        # sum_thread_num_word += len(tokens)
        # sum_thread_num_char += len(text)
        sum_thread_post_depth += post['post_depth']

    # find duplicate posts without body
    # for id in duplicate_id:
    #     if id not in thread['posts']:
    #         print("lost", id)

    thread['num_post'] = len(thread['posts'])
    # thread['avg_num_sent'] = sum_thread_num_sent / thread['num_post']
    # thread['avg_num_word'] = sum_thread_num_word / thread['num_post']
    # thread['avg_num_char'] = sum_thread_num_char / thread['num_post']
    # thread['avg_post_depth'] = sum_thread_post_depth / thread[
    #     'num_post']
    # post depth normalized by the number of comments in the discussion
    for post in thread['posts']:
        # avoid division by 0
        post['post_depth_normalized'] = \
            float(post['post_depth']) / (sum_thread_post_depth + 1)
        # add common thread features

        post_list.append(post)

    thread_list.append(thread)

# post depth normalized by the number of comments in the discussion


# new features
# is_initial_author: True if current author == author of initial post
for thread in thread_list:
    if 'is_self_post' not in thread:
        thread['is_self_post'] = False
    # total number of comments
    thread['num_comments'] = thread['num_post'] - 1
    if (thread['num_comments']) < 0:
        exit(-1)
    # number of unique branches in the discussion tree
    num_branches = 1
    for post in thread['posts']:
        if 'in_reply_to' in post:
            if post['in_reply_to'] == thread['id']:
                num_branches += 1
    thread['num_branches'] = num_branches
    # print(num_branches)
    # average length of all the branches of discussion in the discussion tree
    thread['avg_len_branches'] = \
        (thread['num_comments'] + 1) * 1.0 / num_branches
    posts = thread['posts']
    # check all initial post are in posts[0]
    # if posts[0]['id'] != posts[0]['thread_id']:
    #     print("error")
    #     exit(-1)
    for post in posts:
        if post['is_first_post']:
            post['is_initial_author'] = True
        else:
            if 'author' in post and 'author' in posts[0]:
                post['is_initial_author'] = post['author'] == posts[0][
                    'author']

# is_parent_author:  True if current author == author of parent post
for thread in thread_list:
    posts = thread['posts']
    for post in posts:
        if 'author' not in post:
            continue
        if 'in_reply_to' in post:
            parent_post = None
            for other_post in posts:
                if other_post['id'] == post['in_reply_to']:
                    parent_post = other_post
                    continue
            # if parent_post is None:
            #     print("didn't find parent post")
            #     print(post['id'])
            #     print(post['thread_id'])
            #     exit(-1)
            if parent_post is not None and 'author' in parent_post:
                post['is_parent_author'] = post['author'] == parent_post[
                    'author']
        else:
            # current post is initial post of thread
            post['is_parent_author'] = False

pickle.dump(post_list, open("../cache/post_list.pickle", "wb"))
pickle.dump(thread_list, open("../cache/thread_list.pickle", "wb"))

# add lost comments found from Google BigQuery
all_found = []
with open("../cache/lost_comments.json") as data_file:
    for line in data_file.readlines():
        post = json.loads(line)
        all_found.append(post)
data_file.close()

# add lost posts found from Google BigQuery(all found)
with open("../cache/lost_posts.json") as data_file:
    for line in data_file.readlines():
        post = json.loads(line)
        all_found.append(post)
data_file.close()

all_found_name = set()
for found in all_found:
    all_found_name.add(found['name'])

still_lost = set()


def get_found(id):
    for found in all_found:
        if found['name'] == id:
            return found


for post in post_list:
    if 'body' not in post:
        # comment
        if post['id'].startswith('t1'):
            if post['id'] in all_found_name:
                found = get_found(post['id'])
                post['body'] = found['body']
                # post['text'] = post['body']
                post['author'] = found['author']
            else:
                still_lost.add(post['id'])
        # post
        else:
            if post['id'] in all_found_name:
                found = get_found(post['id'])
                post['title'] = found['title']
                post['body'] = found['selftext']
                post['author'] = found['author']
                # post['text'] = post['title'] + " " + post['body']

print("still lost: ", len(still_lost))

# with open("../cache/still_lost.txt", "w") as file:
#     for i in still_lost:
#         file.write(i + "\n")

# add title to every post
for post in post_list:
    if 'title' not in post:
        post['title'] = ""
# pickle.dump(post_list, open("../cache/post_list.pickle", "wb"))
# pickle.dump(thread_list, open("../cache/thread_list.pickle", "wb"))

import pandas as pd


post_df = pd.DataFrame(post_list)
thread_df = pd.DataFrame(thread_list)

post_df = post_df.set_index('id')
thread_df = thread_df.set_index('id')

# post_df['thread_is_self_post'] = None
# post_df['thread_avg_num_sent'] = None
# post_df['thread_avg_num_word'] = None
# post_df['thread_avg_num_char'] = None
# post_df['thread_avg_num_post_depth'] = None

# # Thread features
# post_df['thread_comment_num'] = None
# post_df['thread_branch_num'] = None
# post_df['thread_branch_len'] = None

# # Structure features
# post_df['parent_num_sent'] = None
# post_df['parent_num_word'] = None
# post_df['parent_num_char'] = None
post_df['parent_post_depth'] = None
post_df['parent_post_depth_normalized'] = None

for index, row in post_df.iterrows():
    # Thread features
    post_df.set_value(index, 'thread_is_self_post',
                      thread_df.loc[row.thread_id].is_self_post)
    post_df.set_value(index, 'thread_comment_num',
                      thread_df.loc[row.thread_id].num_comments)
    post_df.set_value(index, 'thread_branch_num',
                      thread_df.loc[row.thread_id].num_branches)
    post_df.set_value(index, 'thread_branch_len',
                      thread_df.loc[row.thread_id].avg_len_branches)

    # Structure features
    # set to 0 if comment doesn't have a comment

    parent_id = row.in_reply_to
    # post_df.set_value(
    #     index, 'parent_num_sent',
    #     0 if parent_id != parent_id or parent_id not in post_df.index else
    #     post_df.loc[parent_id].num_sent)
    # post_df.set_value(
    #     index, 'parent_num_word',
    #     0 if parent_id != parent_id or parent_id not in post_df.index else
    #     post_df.loc[parent_id].num_word)
    # post_df.set_value(
    #     index, 'parent_num_char',
    #     0 if parent_id != parent_id or parent_id not in post_df.index else
    #     post_df.loc[parent_id].num_char)
    post_df.set_value(
        index, 'parent_post_depth',
        0 if parent_id != parent_id or parent_id not in post_df.index else
        post_df.loc[parent_id].post_depth)
    post_df.set_value(
        index, 'parent_post_depth_normalized',
        0 if parent_id != parent_id or parent_id not in post_df.index else
        post_df.loc[parent_id].post_depth_normalized)

    # post_df.set_value(index, 'thread_avg_num_sent',
    #                   thread_df.loc[row.thread_id].avg_num_sent)
    # post_df.set_value(index, 'thread_avg_num_word',
    #                   thread_df.loc[row.thread_id].avg_num_word)
    # post_df.set_value(index, 'thread_avg_num_char',
    #                   thread_df.loc[row.thread_id].avg_num_char)

    # if row.is_first_post:
    #     title = thread_df.loc[row.thread_id].title
    #     post_df.set_value(index, 'new_title', title)
    #     if row.body == row.body:
    #         post_df.set_value(index, 'new_text',
    #                           str(title) + " " + str(row.body))
    # elif row.body == row.body:
    #         post_df.set_value(index, 'new_text', row.body)

    # test of dataframe
    # post_list = post_list[:1000]


data_dir = "./original/cache/"
post_df_path = data_dir + "post_df.json"
thread_df_path = data_dir + "thread_df.json"
post_df.to_json(path_or_buf=post_df_path, orient='records')
thread_df.to_json(path_or_buf=thread_df_path, orient='records')

# add text of the link in reply to
# none if is initial post
for index, row in post_df.iterrows():
    # Thread features
    parent_id = row.in_reply_to
    # post_df.set_value(
    #     index, 'parent_text',
    #     "" if parent_id != parent_id or parent_id not in
    #     post_df.index
    #     else
    #     post_df.loc[parent_id].text)
    post_df.set_value(
        index, 'parent_title',
        "" if parent_id != parent_id or parent_id not in
        post_df.index
        else
        post_df.loc[parent_id].title)
    post_df.set_value(
        index, 'parent_body',
        "" if parent_id != parent_id or parent_id not in
        post_df.index
        else
        post_df.loc[parent_id].body)

# label
post_df=post_df[post_df.majority_type.notnull()]
post_df = post_df[post_df.majority_type!='other']

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
post_df['label'] = label_encoder.fit_transform(post_df.majority_type)
post_df = post_df.assign(label=post_df.label)


post_df_path = "data.json"
post_df.to_json(path_or_buf=post_df_path, orient='records')


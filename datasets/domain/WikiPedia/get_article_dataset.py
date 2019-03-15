import make_wiki_adjacencies
import graph_manip
import pandas as pd
import numpy as np
import pickle as pkl
import json


def convert_title(title):
    # FIXME: something weird going on with binary characters? u'\x81'? just converting it to string and hoping it's ok for now
    '''
    newtitle = ""
    for character in title:
        newtitle += chr(ord(character))
    return newtitle.replace("_"," ")
    '''
    return title


def get_cluster_articles(G, categories):
    articles = []
    for category in categories:
        if category in G.nodes():
            for neighbor in G.neighbors(category):
                if neighbor[:9] != "Category:":
                    topics = [category for category in G.neighbors(neighbor) if category[:9] == "Category:"]
                    articles.append([neighbor, topics])
                    if len(articles) % 100 == 0:
                        print("    number of articles: %i, last article topics: %s" % (len(articles), str(topics)))
    return articles


def partition_dataset(rows, fraction_train, fraction_dev):
    if type(rows) == str:
        with open(rows, "r") as rowsfile:
            df = pd.read_csv(dataset)
            rows = [[row[1], row[2]] for i, row in df.iterrows()]
    np.random.shuffle(rows)
    # separate into train and test
    numtrain = int(len(rows) * fraction_train)
    numdev = int(len(rows) * fraction_dev)
    train_rows = rows[:numtrain]
    dev_rows = rows[numtrain:numtrain + numdev]
    test_rows = rows[numtrain + numdev:]

    return train_rows, dev_rows, test_rows


def main(article_categories_file, cluster_groupings_file, cluster_names_file, dataset_file, fraction_train=.7,
         fraction_dev=.15):
    # attach articles to category graph
    G = graph_manip.Graph()
    make_wiki_adjacencies.add_adjacencies(G, article_categories_file)

    # get cluster_names, cluster_categories from cluster_groupings_file

    with open(cluster_groupings_file, 'r') as clustergroups:
        clusters_categories = pkl.load(clustergroups)

    with open(cluster_names_file, 'r') as clusternames:
        clusters_names = pkl.load(clusternames)

    rows = []

    # write out articles and labels sorted randomly and separated into dev and training set
    # note that the format of the dataset is: [{"text":<article>,"label":<category title>},...]
    for i, (cluster_id, cluster_categories) in enumerate(clusters_categories.items()):
        if i > 44: break
        articles = get_cluster_articles(G, cluster_categories)
        for article in articles:
            rows.append({"text": str(article[0]), "label": clusters_names[cluster_id], "labels": article[1]})
        if ((i + 1) % 1) == 0:
            print("there are " + str(len(articles)) + " articles in cluster \"" + clusters_names[cluster_id] + "\"")
            print(str(i + 1) + " / " + str(len(clusters_categories)))

    np.random.shuffle(rows)

    train_rows, dev_rows, test_rows = partition_dataset(rows, fraction_train, fraction_dev)

    with open(dataset_file + "_train.json", 'w') as f:
        json.dump(train_rows, f)
    with open(dataset_file + "_dev.json", 'w') as f:
        json.dump(dev_rows, f)
    with open(dataset_file + "_test.json", 'w') as f:
        json.dump(test_rows, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("article_categories_file")
    parser.add_argument("cluster_groupings_file")
    parser.add_argument("cluster_names_file")
    parser.add_argument("dataset_file")

    args = parser.parse_args()

    main(args.article_categories_file, args.cluster_groupings_file, args.cluster_names_file, args.dataset_file)

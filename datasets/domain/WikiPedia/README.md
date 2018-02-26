# Wikipedia Clustering Dataset

Here is a series of scripts for creating a dataset of clustered wikipedia articles from the network structure using DBPedia datasets.

## Datasets from DBPedia

1. English skos categories file (with the ttl extension) - used to construct a graph of the connections between Wikipedia category pages
2. Article categories file (with the ttl extension) - used to link the articles to their categories
3. Wikipedia XML source dump file (with the xml extension) - used to get the articles in Wikipedia

## Scripts

1. make_wiki_adjacencies.py - constructs graph for all categories in wikipedia and writes it to a file

		* Input: the skos categories file

		* Output: adjacencies.txt (enumerates nodes followed by adjacency list)

2. graph_manip.py

3. get_categories*.py - There are a few different versions of this script.  Each version has the same inputs and outputs.

		* Input: adjacencies.txt - the adjacencies file made from make_wiki_adjacencies script

		* Outputs: cluster_groupings.pkl (enumerates categories for each cluster), cluster_names.pkl (enumerates a name for each cluster)

4. get_article_dataset.py

		* Inputs: cluster_groupings.pkl, cluster_names.pkl

		* Output: dataset_[train/dev/test].json

5. title_to_text.py

                * Inputs: dataset_[train/dev/test].json

                * Output: dataset_[train/dev/test]_content.json


## Creating Dataset

### Pre-requisites

-gensim
-networkx
-pandas

### Running the scripts

python make_wiki_adjacencies.py [-h] [-e EMBEDDINGS_FILE] [-s STOPWORDS_FILE]
                                skos_file adjacencies_file

python get_categories3.py [-h] [-d]
                          adjacencies_file cluster_groupings_file
                          cluster_names_file

python get_article_dataset.py [-h]
                              article_categories_file cluster_groupings_file
                              cluster_names_file dataset_file

python title_to_text.py [-h] wiki_dump_file dataset_files

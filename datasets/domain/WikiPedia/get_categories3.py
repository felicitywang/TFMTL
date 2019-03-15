import graph_manip
import make_wiki_adjacencies


def main(adjacencies_file, cluster_groupings_file, cluster_names_file,
         from_dbpedia=False):
    number_of_clusters = 100
    topological_sorting_file = None
    # topological_sorting_file = "temp/topological_sorting.txt"
    root_node = "Category:Main_topic_classifications"
    top_depth = 1
    cluster_depth = 2

    if from_dbpedia:
        G = make_wiki_adjacencies.add_adjacencies(graph_manip.DiGraph(),
                                                  adjacencies_file)
    else:
        G = graph_manip.load_graph(adjacencies_file)
    cluster_groupings = {}
    cluster_names = {}
    cluster_tops = graph_manip.get_n_level_graph_from(G, root_node, top_depth,
                                                      leaf_nodes=True)

    for i, node in enumerate(cluster_tops):
        # cluster_groupings[i] = [node]
        cluster_groupings[str(i)] = list(
            graph_manip.get_n_level_graph_from(G, node, cluster_depth).nodes())
        cluster_names[str(i)] = node

    graph_manip.write_clusters(cluster_groupings_file, cluster_groupings,
                               cluster_names_file, cluster_names,
                               verbose=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("adjacencies_file")
    parser.add_argument("cluster_groupings_file")
    parser.add_argument("cluster_names_file")
    parser.add_argument("-d", "--from_dbpedia", action="store_true")

    args = parser.parse_args()

    main(args.adjacencies_file, args.cluster_groupings_file,
         args.cluster_names_file, from_dbpedia=args.from_dbpedia)

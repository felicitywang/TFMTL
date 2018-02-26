from networkx import *
import pickle as pkl

def remove_nodes_if(G, should_remove):
	usefull_nodes = []
	for node in nodes_iter(G):
		if not should_remove(node):
			usefull_nodes.append(node)
	return subgraph(G, usefull_nodes)

def trim_graph_nodes(G, iterations=0, useless_cuttoff=0, root=None):
	if root or iterations:
		print("trimming graph nodes")
		print("\tcopying graph")
		new_graph = G
		if root:
			print("\tgetting subgraph consisting of descendants of "+root)
			new_graph = subgraph(new_graph, descendants(new_graph, root))
			print("\tnow has "+str(len(new_graph))+" nodes")
		if iterations:
			print("\ttrimming leaves")
			for i in range(iterations):
				initial_length = len(new_graph)
				new_graph = remove_nodes_if(new_graph, lambda x: len(new_graph.neighbors(x)) <= useless_cuttoff)
				print("\t\tremoved "+str(initial_length-len(new_graph))+" nodes")
				print("\t\t"+str(i+1)+" / "+str(iterations))
				if (initial_length-len(new_graph)) == 0:
					print("\tstopping iterations early")
					break
		print("done trimming graph nodes")
		return new_graph
	else:
		return G

#don't keep subgraphs that are less than n
#only keep top p of nodes in each subgraph
def trim_subgraphs(components, subgraph_cuttoff=None, n_top_nodes=None):
	print("trimming subgraphs")

	#cut unnecessary subgraphs
	new_components = []
	if subgraph_cuttoff:
		print("\tcutting subgraphs with under "+str(subgraph_cuttoff)+" nodes")
		print("\t\tinitial number of components: "+str(len(components)))
		for component in components:
			if len(component) > subgraph_cuttoff:
				print("\t\tadding component which has "+str(len(component))+" nodes")
				new_components.append(component)
		print("\t\tfinal number of components: "+str(len(new_components)))
	else:
		new_components = components

	if n_top_nodes:
		#get rid of cycles in remaining components
		print("\tmaking components into acyclic graphs")
		for i,component in enumerate(new_components):
			print("\t\tfinding cycles in graph with "+str(len(component))+" nodes")
			cycles = simple_cycles(component)
			print("\t\t"+str(len(list(cycles)))+" cycles found")
			for cycle in cycles:
				#print(cycle)
				if len(cycle) > 1:
					u, v = cycle[0], cycle[1]
				else:
					u, v = cycle[0], cycle[0]
				try:
					component.remove_edge(u, v)
				except exception.NetworkXError as e:
					print(e)
					continue
			print("\t\t"+str(i+1)+" / "+str(len(new_components)))

		#topologically sort remaining components and
		#get rid of all nodes except for top p
		print("\ttopologically sorting and cutting bottom nodes on the graphs")
		for component in new_components:
			component.remove_nodes_from(topological_sort(component)[n_top_nodes:])

		#expand components into not connected subgraphs and return final components
		print("\texpanding graphs into their connected subgraphs")
		new_components2 = []
		for component in new_components:
			new_components2.extend(weakly_connected_components(component))
		new_components = [subgraph(G, list(vertices)) for vertices in new_components2]
	new_components = sorted(new_components, key=len)
	print("done trimming subgraph")
	return new_components

def print_graph_info(G,verbose=False):
	### PRINT INFORMATION ABOUT GRAPH ###
	n_nodes = len(G.nodes())
	n_edges = len(G.edges())
	print("number of nodes: "+str(n_nodes)+", total number of edges: "+str(n_edges))
	if verbose:
		print("directed acyclic: "+str(is_directed_acyclic_graph(G)))
		print("weakly connected: "+str(is_weakly_connected(G)))

def load_graph(adjacencies_file,verbose=False):
	G = DiGraph()
	print("loading from adjacencies file ("+adjacencies_file+")")
	with open(adjacencies_file,"r") as adjfile:
		for i,line in enumerate(adjfile):
			line = line.strip()
			node = line.split()[0]
			nodes = eval(line[len(node)+1:])
			nodenames = [n[0] for n in nodes]
			nodenames.append(node)
			G.add_nodes_from(nodenames)
			G.add_edges_from([(node,n[0],{"weight":float(n[1])}) for n in nodes])
			if (i+1) % 100000 == 0: print(i+1)
	print("done loading graph")

	print("INFORMATION ABOUT FULL GRAPH")
	print_graph_info(G,verbose=verbose)
	return G

def get_center_node(G):
	centralities = betweenness_centrality(G.to_undirected())
	node_centrality = max(((node,centrality) for node,centrality in centralities.items()), key=lambda x:x[1])
	return node_centrality[0]

def write_clusters(cluster_groupings_file, cluster_groupings, cluster_names_file, cluster_names, verbose=False):
	print("writing "+str(len(cluster_groupings))+" clusters to files.")
	if verbose:
		print("The groups contain the following numbers of categories respectively: "+str([len(group) for cluster,group in cluster_groupings.items()]))
	with open(cluster_groupings_file, "w") as groupfile:
		'''
		for cluster, nodes in cluster_groupings.items():
			groupfile.write(str(cluster)+" "+str(nodes)+"\n")
		'''
		pkl.dump(cluster_groupings, groupfile)

	with open(cluster_names_file, "w") as namesfile:
		'''
		for cluster, name in cluster_names.items():
			namesfile.write(str(cluster)+" "+name+"\n")
		'''
		pkl.dump(cluster_names, namesfile)
	print("done writing clusters to files")

def compute_clustering(original_graph, components, number_of_clusters, cluster_groupings_file, cluster_names_file, topological_sorting_file=None):
	print("computing clustering")
	from sklearn import cluster as cl
	clusters = []
	topsort = []
	n_nodes = sum(len(component) for component in components)
	for i,graph in enumerate(components):
		print("\nthis is a "+str(len(graph))+"-node graph")
		is_acyclic = is_directed_acyclic_graph(graph)
		print("directed acyclic: "+str(is_acyclic))
		if topological_sorting_file and is_acyclic:
			topsort.append(topological_sort(graph))
			print("\tdone with topological sort")
			print("\ttop 3 categories: "+str(topsort[-1][:3]))
		#print(m.todense())
		n = int(round(number_of_clusters*(float(len(graph))/float(n_nodes))))
		if n < 1:
			print("\tthis is 1 cluster")
			clusters.append(list(zip(graph.nodes(), [0]*len(graph))))
		else:
			print("\tdoing spectral clustering to produce "+str(n)+" clusters")
			#if len(graph) > 100000: print("jk, not gonna do this now"); continue
			print("\t\tcreating adjacency matrix")
			tempnodes = graph.nodes()
			m = adjacency_matrix(graph.to_undirected(),nodelist=tempnodes)
			print("\t\tdone creating adjacency matrix")
			print("\t\tstarting clustering")
			#sp = cl.SpectralClustering(n_clusters=n, n_jobs=-1, affinity="precomputed")
			sp = cl.SpectralClustering(n_clusters=n, affinity="precomputed")
			clusters.append(list(zip(tempnodes, sp.fit_predict(m))))
		print(str(i+1)+" / "+str(len(components)))
	print("done clustering")

	print("grouping and creating cluster ids")
	#print(clusters)
	graphclusternum = 0
	cluster_groupings = {}
	#cluster_mappings = []
	for graphcluster in clusters:
		for node, cluster in graphcluster:
			if str(graphclusternum)+" "+str(cluster) not in cluster_groupings.keys():
				cluster_groupings[str(graphclusternum)+" "+str(cluster)] = []
			cluster_groupings[str(graphclusternum)+" "+str(cluster)].append(node)
			#cluster_mappings.append((node,str(graphclusternum)+" "+str(cluster)))
		graphclusternum += 1
	cluster_names = {}
	for clusterkey,nodes in cluster_groupings.items():
		cluster_names[clusterkey] = get_center_node(subgraph(original_graph, nodes))[9:]
	#print(cluster_mappings)
	#print(cluster_groupings)

	### WRITE TO FILES ###
	print("writing clusters to files")
	write_clusters(cluster_groupings_file, cluster_groupings, cluster_names_file, cluster_names)

	if topological_sorting_file:
		with open(topological_sorting_file, "w") as topfile:
			for nodelist in topsort:
				for node in nodelist:
					topfile.write(node+"\n")
				topfile.write("\n")

def get_n_level_graph_from(original_graph, root, n, leaf_nodes=False):
	prev_nodes = [root]
	saved_nodes = set([root])
	for i in range(n):
		neighbors = []
		for node in prev_nodes:
			neighbors.extend(original_graph.neighbors(node))
		saved_nodes = saved_nodes.union(neighbors)
		prev_nodes = neighbors
		print(str(len(saved_nodes))+" nodes")
		print(str(i+1)+" / "+str(n))
	if leaf_nodes:
		return prev_nodes
	else:
		return subgraph(original_graph, saved_nodes)

def choose_nodes(nodes, n_nodes, distribution=None):
	if distribution == None:#uniform distribution
		np.random.shuffle(nodes)
		return nodes[:n_nodes]
	else:#use distrubtion specified
		raise NotImplementedError

def sample_graph(original_graph, n_nodes):
	if n_nodes > len(original_graph.nodes()):
		print("Warning: there are only "+str(len(original_graph.nodes()))+" nodes in the graph, not "+str(n_nodes)+", so returning whole graph")
		return original_graph
	nodes = list(original_graph.nodes())
	
	nodes = choose_nodes(nodes, n_nodes)
	new_graph = Graph()
	for node1 in nodes:
		new_graph.add_node(node1)
		for node2 in nodes:
			if node2 == node1: continue
			new_graph.add_node(node2)
			path = shortest_path(G,source=node1,target=node2)
			new_weight = mul(G[node][path[i+1]]["weight"] for i,node in enumerate(path[:-1]))
			new_graph.add_edge(node1,node2,weight=new_weight)
	return new_graph



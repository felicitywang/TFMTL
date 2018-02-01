import numpy as np
from networkx import *

def get_embeddings(glovefile):
	print("getting embeddings")
	embeddings = {}
	with open(glovefile, 'r') as embeddings_file:
		for line in embeddings_file:
			linesplit = line.split()
			word = linesplit[0]
			values = [float(numstr) for numstr in linesplit[1:]]
			embeddings[word] = values
	print("done getting embeddings!")

	return embeddings

#takes in words and embeddings and calculates the final vector embedding (as average) and returns it in a numpy array
def get_embedding_rep(words, embeddings):
	final_vector = np.zeros(len(embeddings[embeddings.keys()[0]]))
	#print(final_vector)
	num_embedded_words = 0
	for i,word in enumerate(words):
		if word.lower() in embeddings.keys():
			final_vector = np.add(final_vector, np.array(embeddings[word.lower()]))
			num_embedded_words += 1
		else:
			print(word)
		#print(str(i)+" / "+str(len(words)), word)
	if num_embedded_words != 0:
		final_vector /= num_embedded_words
	else:
		#print("Warning: returning an array of zeros because none of the words have embeddings!")
		print("Warning: none of the words have embeddings!")
		return None

	return final_vector

def get_stop_words(stopwords_file):
	dumbwords = []
	if stopwords_file:
		with open(stopwords_file, 'r') as stopwords:
			for line in stopwords:
				if line[0] != "#":
					dumbwords.append(line.strip())
	return dumbwords

def get_cosine_sim(rep1, rep2):
	return np.divide(float(np.dot(rep1,rep2)),(np.linalg.norm(rep1)*np.linalg.norm(rep2)))

def add_adjacencies(G,input_filename,embeddings_file=None,stopwords_file=None,only_attached=False):
	print("loading adjacencies from dbpedia file ("+input_filename+")")
	import re
	beginningstr = "<http://dbpedia.org/resource/"
	endingstr = ">"
	pattern = re.compile(beginningstr+".*?"+endingstr)
	if embeddings_file:
		dumbwords = get_stop_words(stopwords_file)
		embeddings = get_embeddings(embeddings_file)
	with open(input_filename, "r") as infile:
		for i,line in enumerate(infile):
			#if i > 1000: break
			line = line.strip()
			if line[0] == "#": continue
			splitline = line.split()
			#print(splitline[0],splitline[2])
			page1 = re.findall(pattern,splitline[0])
			page2 = re.findall(pattern,splitline[2])
			if len(page1) == 0 or len(page2) == 0:
				if (i+1) % 100000 == 0: print(i+1)
				continue
			page1 = page1[0].replace(beginningstr,"").replace(endingstr,"")
			page2 = page2[0].replace(beginningstr,"").replace(endingstr,"")
			#print(page1,page2)
			if only_attached and (page1 not in G.nodes()) and (page2 not in G.nodes()):
				if (i+1) % 100000 == 0: print(i+1)
				continue
			G.add_node(page1)
			G.add_node(page2)
			if embeddings_file:
				#FIXME: need to figure out a way to handle Nan, "Trek:", "Anti-blabla", and "(Caribbean)", "Kerby's", "Metalogic"
				#also, this is extremely slow
				words1 = [word for word in page1[9:].split("_") if word not in dumbwords]
				words2 = [word for word in page2[9:].split("_") if word not in dumbwords]
				#print(len(words1),len(words2))
				rep1 = get_embedding_rep(words1,embeddings)
				rep2 = get_embedding_rep(words2,embeddings)
				#FIXME: is this the right move?
				if type(rep1) == type(None):
					G.remove_node(page1)
					continue
				if type(rep2) == type(None):
					G.remove_node(page1)
					continue
				cosine_sim = get_cosine_sim(rep1, rep2)

				G.add_edge(page2,page1,weight=cosine_sim)
			else:
				G.add_edge(page2,page1,weight=1.0)
			if (i+1) % 100000 == 0: print(i+1)
	print("done adding adjacencies")

# write the adjacency list to file
def write_adjacency_file(G, adjacencies_file):
	print("writing adjacencies to file ("+adjacencies_file+")")
	with open(adjacencies_file,"w") as adjfile:
		for node in G.nodes():
			adjfile.write(node+" "+str([(n, G.edge[node][n]["weight"]) for n in G.neighbors(node)])+"\n")
	print("done writing to file")

'''
"/Users/jeredmcinerney/Desktop/Intelellectual/glove/glove.6B.300d.txt"
'stopwords.txt'
'''

def main(skos_file, adjacencies_file, embeddings_file=None, stopwords_file=None):
	G = DiGraph()
	add_adjacencies(G, skos_file,embeddings_file=embeddings_file,stopwords_file=stopwords_file)
	write_adjacency_file(G, adjacencies_file)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()

	parser.add_argument("skos_file")
	parser.add_argument("adjacencies_file")
	parser.add_argument("-e", "--embeddings_file",type=str,default=None)
	parser.add_argument("-s", "--stopwords_file",type=str,default=None)

	args = parser.parse_args()

	main(args.skos_file, args.adjacencies_file, embeddings_file=args.embeddings_file, stopwords_file=args.stopwords_file)

from gensim.corpora.wikicorpus import extract_pages
import argparse
import json
import bz2

def main(wiki_dump_file, data):
    indices = {example["text"]:i for i,example in enumerate(data)}
    print(len(data))
    print(data[0])
    num_replaced = 0
    with bz2.BZ2File(wiki_dump_file) as f:
        for i,(title, content, pageid) in enumerate(extract_pages(f)):
            try:
                data[indices[title]]["text"] = content
                data[indices[title]]["replaced"] = True
                if num_replaced % 100 == 0:
                    print(content[:100]+"...\n")
                num_replaced += 1
            except Exception:
                pass
            if i % 1000 == 0:
                #print(title)
                #print(content)
                #print(pageid)
                print("%i of %i, %i" % (num_replaced, len(data), i))

    return data

def cleanup(data):
    new_data = []
    for example in data:
        if "replaced" in example.keys():
            del example["replaced"]
            new_data.append(example)
    return new_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("wiki_dump_file")
    parser.add_argument("dataset_files")
    args = parser.parse_args()

    data = []
    dataset_files = args.dataset_files.split(",")
    print(dataset_files)

    indices = [0]
    for dataset_file in dataset_files:
        with open(dataset_file, 'r') as f:
            data.extend(json.load(f))
            indices.append(len(data))
    print(indices)

    data = main(args.wiki_dump_file, data)

    for i,dataset_file in enumerate(dataset_files):
        with open(dataset_file[:-5]+"_content.json", 'w') as f:
            json.dump(cleanup(data[indices[i]:indices[i+1]]), f)

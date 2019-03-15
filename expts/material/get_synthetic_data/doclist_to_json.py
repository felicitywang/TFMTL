# convert a file of docuemets to .json format.
# the docuements are identified by "doc_id:"
# doclist_to_json.py <doc_file> <label> <output_json_file>
# example: python3 doclist_to_json.py random5.docs 0

import codecs
import json
import sys

docs_file = open(sys.argv[1])  # random5.docs
concat = ""
docs_list = []
line = docs_file.readline()
line = line.replace('\n', '')
doc_id = line[8:]
for line in docs_file:
    line = line.replace('\n', '')
    if line.startswith("doc_id:"):
        concat = concat[:len(concat) - 13]  # trim last "REPLACEWITHR "
        docs_list.append({'doc_id': doc_id, 'text': concat, 'label': sys.argv[2]})
        concat = ""
        doc_id = line[8:]
    else:
        if (line != "" and not line.startswith("tgz_file:")):
            concat = concat + line + "REPLACEWITHR "
concat = concat[:len(concat) - 13]  # trim last "REPLACEWITHR "
docs_list.append({'doc_id': doc_id, 'text': concat, 'label': sys.argv[2]})
with codecs.open(sys.argv[3], 'w', encoding='utf-8') as f:
    json.dump(docs_list, f, ensure_ascii=False)

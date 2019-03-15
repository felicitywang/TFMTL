#!/usr/bin/env bash
# convert a file of docuemets to .json format.
# the docuements are identified by "doc_id:"
# doclist_to_json.py <doc_file> <label> <output_json_file>
# example: sh doclist_to_json.sh doclist.txt 0 data.json

doc_file=$1
label=$2
json_output=$3

python3 doclist_to_json.py $doc_file $label $json_output

# replace \r:
sed -i 's/REPLACEWITHR /\\r /g' data.json

# Get in-domain synthetic data for each domain
# Usage: sh create_syn.sh query_folder domain_name data_length
# e.g. sh create_syn.sh syn_p1000 GOV 1000
# this would read syn_p1000/GOVq as queries for domain GOV and save the synthetic data in current_folder/data/json/GOV_syn_p1000/data.json.gz


# arguments
query_folder="$1"
domain="$2"
length="$3"

dataset_folder="data/json/${domain}_${query_folder}"
mkdir -p $dataset_folder

doc_id_file="${query_folder}/${domain}_${length}_doc_ids.txt"
doc_file="${query_folder}/${domain}_${length}_doc_list.txt"

# echo ${query_folder}
# echo ${domain}
# echo ${length}
# echo ${dataset_folder}

# echo

# Extract Wikipedia doc_ids from CAW using Lucene search tool
# java -cp concrete-lucene-assembly-4.12.1.jar edu.jhu.hlt.concrete.lucene.search.Search <query> <wikipedia_index> <number_of_doc_ids>
java -cp ~tongfei/proj/concrete-lucene/concrete-lucene-assembly-4.12.1.jar edu.jhu.hlt.concrete.lucene.search.Search "${query_folder}/${domain}q" /export/b01/tongfei/exp/caw-lucene $length | awk -F"\t" '{print $3}' > $doc_id_file

# Read doc_ids and extract text from CAW using Concrete Python API
# python random_access_fast.py <doc_ids> <output>
python random_access_fast.py $doc_id_file $doc_file

# Write documents in the json format {'doc_id': , 'text': , 'label': }
# doclist_to_json.py <doc_file> <label> <output_json_file>
sh doclist_to_json.sh $doc_file 1 "data/json/${domain}_${query_folder}/data.json"


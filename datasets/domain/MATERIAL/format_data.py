import json

doc_domain_path = "domain_annotation.tsv"
doc_path_1 = "translation/"

doc_domain_file = open(doc_domain_path)
doc_domain_list = []

next(doc_domain_file)  # skip the first line

for line in doc_domain_file:
    if (".wav" not in line) and (
        "\tD" in line):  # .txt files with non-empty domains
        line = line.replace('\n', '')
        doc_domain = line.split("\t")
        labelD = doc_domain[1]
        if labelD == "D07":
            # label = "Government-And-Politics"
            label = 0
        elif labelD == "D09":
            # label = "Lifestyle"
            label = 1

        file_name = doc_domain[0][:-4]

        doc_path = doc_path_1 + file_name + ".translation.eng.txt"
        doc_file = open(doc_path)

        text = ""
        for doc_line in doc_file:
            doc_line = doc_line.replace('\n', '')
            texts = doc_line.split("\t")
            text = text + " " + texts[2]  # English translation

        doc_domain_list.append(
            {'text': text, 'label': label, 'file_name': file_name})

json.dump(doc_domain_list, open('data.json', 'w'))

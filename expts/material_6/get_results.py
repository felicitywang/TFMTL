import os
datasets = []
results = []
for file_name in os.listdir('./'):
    if not file_name.endswith('log'):
        continue
    with open(file_name) as file:
        for line in file.readlines():
            if line.startswith('Metrics on highest-accuracy epoch for dataset'):
                index = line.find('Acc')
                dataset = line[46:line.find(':')]
                res = line[index+6:index+14]
                datasets.append(dataset)
                results.append(res)

    with open(file_name[:file_name.find('log')]+'res', 'w') as file:
        for dataset in datasets:
            file.write(dataset+', \n')
        for res in results:
            file.write(res+'\n')

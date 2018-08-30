"""Parse the dev results from the server in csv format"""
import os
import sys

DOMAIN_NAMES = {
  'Government-And-Politics': 'GOV',
  'Lifestyle': 'LIF',
  'Business-And-Commerce': 'BUS',
  'Law-And-Order': 'LAW',
  'Sports': 'SPO',
  'Physical-And-Mental-Health': 'HEA',
  'Military': 'MIL',
}


def main():
  dir = sys.argv[1]

  domains = ['BUS', 'GOV-A', 'GOV-B', 'HEA', 'LAW', 'LIF-A', 'LIF-B', 'MIL',
             'SPO-A', 'SPO-B']

  p_misses = dict()
  p_fas = dict()

  for filename in os.listdir(dir):
    if not filename.endswith('.csv'):
      continue
    with open(os.path.join(dir, filename)) as file:
      lines = [line.strip() for line in file.readlines() if
               line.strip()]
      # print(len(lines))
      # print(lines)
      for i in range(int(len(lines) / 3)):
        domain_name = DOMAIN_NAMES[lines[i * 3]]
        if domain_name in ['GOV', 'LIF', 'SPO']:
          if '1A' in filename:
            domain_name += '-A'
          elif '1B' in filename:
            domain_name += '-B'
          else:
            raise ValueError('wrong name ' + filename)
        score_names = lines[i * 3 + 1].split(',')

        assert score_names == [
          'P_truePositive', 'P_miss', 'P_falseAlarm', 'P_trueNegative']

        scores = lines[i * 3 + 2].split(',')

        # print(domain_name)
        # print(scores)

        p_misses[domain_name] = scores[1]
        p_fas[domain_name] = scores[2]

  # print(p_misses)
  # print(p_fas)

  print(' '.join(domains))

  print('p_miss')
  for domain_name in domains:
    print(p_misses[domain_name], end=',')

  print()
  print('p_fa')
  for domain_name in domains:
    print(p_fas[domain_name], end=',')

  print()


if __name__ == '__main__':
  main()

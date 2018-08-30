"""Get write bash scripts for each dataset"""

pos_cuts = [90, 80, 70, 60, 50]
neg_cuts = [50, 50, 50, 50, 50]

raw_dir = 'data/raw/TURK'
json_dir = 'data/json/'

domains = ['GOV', 'LIF', 'BUS', 'LAW', 'HEA', 'MIL', 'SPO']
for pos, neg in zip(pos_cuts, neg_cuts):
  for domain in domains:
    name = 'TURK_' + domain + '_' + str(pos) + '_' + str(neg) + '_ORACLE'
    # print(dir)
    print('python '
          '../scripts/write_tfrecords_single.py ', name,
          ' args_nopretrain.json')


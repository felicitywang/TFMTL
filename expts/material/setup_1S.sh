# GOV
python combine.py --domain GOV --train gold_oracle gold_one_1A_DEV gold_one_1B_DEV gold_one_1A_EVAL --valid gold_oracle_1S_ANALYSIS1

# BUS
python combine.py --domain BUS --train gold_oracle gold_one_1A_DEV gold_one_1A_EVAL

# LAW
python combine.py --domain LAW --train gold_oracle gold_one_1A_DEV gold_one_1A_EVAL

# MIL
python combine.py --domain MIL --train gold_oracle --valid gold_oracle_1S_ANALYSIS1
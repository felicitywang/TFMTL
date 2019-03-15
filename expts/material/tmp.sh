# 410 no pretrain
python get_server_submissions.py submissions/410_gold_only_nopretrain/ DEV one; python get_dev_scores_1s.py submissions/410_gold_only_nopretrain/DEV/one/1S


# 430 glove
python get_server_submissions.py submissions/430_gold_only_pretrained/ DEV one; python get_dev_scores_1s.py submissions/430_gold_only_pretrained/DEV/one/1S


# 420 turk 80 50 nopretrain
python get_server_submissions.py submissions/420_init_gold_only_ft8050_nopretrain/ DEV one; python get_dev_scores_1s.py submissions/420_init_gold_only_ft8050_nopretrain/DEV/one/1S

# 440 turk 80 50 glove
python get_server_submissions.py submissions/440_init_gold_only_ft_8050_glove/ DEV one; python get_dev_scores_1s.py submissions/440_init_gold_only_ft_8050_glove/DEV/one/1S


# 450 7050 nopretrain
python get_server_submissions.py submissions/450_7050_nopretrain/ DEV one; python get_dev_scores_1s.py submissions/450_7050_nopretrain/DEV/one/1S


# 460 70 50 glove
python get_server_submissions.py submissions/460_7050_glove/ DEV one; python get_dev_scores_1s.py submissions/460_7050_glove/DEV/one/1S

# 470 50 50 nopretrain
python get_server_submissions.py submissions/470_5050_nopretrain DEV one; python get_dev_scores_1s.py submissions/470_5050_nopretrain/DEV/one/1S

# 480 50 50 glove
python get_server_submissions.py submissions/480_5050_glove/ DEV one; python get_dev_scores_1s.py submissions/480_5050_glove/DEV/one/1S

# primary
python get_server_submissions.py submissions/p DEV one; python get_dev_scores_1s.py submissions/p/DEV/one/1S

# contrastive 1
python get_server_submissions.py submissions/c1 DEV one; python get_dev_scores_1s.py submissions/c1/DEV/one/1S

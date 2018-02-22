**Sogou News Topic Classification Dataset**

**Summary**

The SogouCA and SogouCS Chinese news dataset is collected by SogouLab from various news website. The Sogou news topic classification dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu) from a combination of SogouCA and SogouCS.

The Sogou news topic classification dataset is constructed by manually labeling each news article according to its URL, which represents roughly the categorization of news in their websites. 5 largest categories are chosen for the dataset. The Pinyin texts are converted using pypinyin combined with jieba Chinese segmentation system.

**Basic stats:**
+ \# items = 510,000
+ \# labels = 5
    - sports
    - finance
    - entertainment
    - automobile
    - technology


**Basic Unit**: document

** Split: train : test = 450,000 : 60,000 **

**bibtex**

```
@article{DBLP:journals/corr/ZhangZL15,
  author    = {Xiang Zhang and
               Junbo Jake Zhao and
               Yann LeCun},
  title     = {Character-level Convolutional Networks for Text Classification},
  journal   = {CoRR},
  volume    = {abs/1509.01626},
  year      = {2015},
  url       = {http://arxiv.org/abs/1509.01626},
  archivePrefix = {arXiv},
  eprint    = {1509.01626},
  timestamp = {Wed, 07 Jun 2017 14:41:26 +0200},
  biburl    = {http://dblp.org/rec/bib/journals/corr/ZhangZL15},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```

[**Webpage**](http://www.sogou.com/labs/dl/ca.html)

[**Webpage**](http://www.sogou.com/labs/dl/cs.html)

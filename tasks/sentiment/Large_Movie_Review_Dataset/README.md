**Large Movie Review Datset**

**Summary**

Labeled and unlabeled IMDB movie reviews with sentiment polarities. There are no neutral reviews in the labeled train/test sets(only positive ones with scores >= 7 and negative ones with scores <= 4) but reviews of any rating are included in the unsupervised set.

**Basic stats:**

+ \# items = 50,000(train 25,000, test 25,000) (unlabeled: 50,000)

+ \# labels = 2(pos/neg)

**Basic Unit**: document

**bibtex**

```
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```

[**Webpage**](http://ai.stanford.edu/~amaas/data/sentiment/)




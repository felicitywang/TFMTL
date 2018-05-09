**Stance -- SemEval-2016 Task 6*

**Summary**

Stance detection here means automatically determining from text whether the author is in favor of the given target, against the given target, or whether neither inference is likely. A tweet and its target(may not appear in the tweet) are given, along with the opinion towards the target and the sentiment of the tweet.

This dataset is a sub-set of the SemEval 2016 Task 6. It's used by arXiv:1802.09913. The training set consists of all the original training data except those with target "Hillary Clinton", dev set all the training data with target "Hillary Clinton", and test set all the test data with target "Donald Trump".

**Basic stats:**

+ \# items = 4,900
+ \# labels = 3
    - 0: AGAINST
    - 1: FAVOR
    - 2: NONE

**Basic Unit**: sentence

**Split**: train : dev : test = 3209 : 984 : 707

**bibtex**
```
@inproceedings{mohammad2016semeval,
  title={Semeval-2016 task 6: Detecting stance in tweets},
  author={Mohammad, Saif and Kiritchenko, Svetlana and Sobhani, Parinaz and Zhu, Xiaodan and Cherry, Colin},
  booktitle={Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016)},
  pages={31--41},
  year={2016}
}

@article{mohammad2017stance,
  title={Stance and sentiment in tweets},
  author={Mohammad, Saif M and Sobhani, Parinaz and Kiritchenko, Svetlana},
  journal={ACM Transactions on Internet Technology (TOIT)},
  volume={17},
  number={3},
  pages={26},
  year={2017},
  publisher={ACM}
}

@inproceedings{sobhani2016detecting,
  title={Detecting stance in tweets and analyzing its interaction with sentiment},
  author={Sobhani, Parinaz and Mohammad, Saif and Kiritchenko, Svetlana},
  booktitle={Proceedings of the Fifth Joint Conference on Lexical and Computational Semantics},
  pages={159--169},
  year={2016}
}

@article{augenstein2016stance,
  title={Stance detection with bidirectional conditional encoding},
  author={Augenstein, Isabelle and Rockt{\"a}schel, Tim and Vlachos, Andreas and Bontcheva, Kalina},
  journal={arXiv preprint arXiv:1606.05464},
  year={2016}
}

arXiv:1802.09913

```

[**Webpage**](http://alt.qcri.org/semeval2016/task6/)

[arXiv:1802.09913](https://arxiv.org/abs/1802.09913)
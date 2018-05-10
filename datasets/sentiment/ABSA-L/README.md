**ABSA-L -- Aspect-based Sentiment Analysis - Laptops**

**Summary**

Aspect-based sentiment analysis is the task of identifying whether an aspect, i.e. a particular property of an item is associated with a positive, negative, or neutral sentiment.


This dataset is a sub-set of the SemEval 2016 Task 5 Subtask 1 Slot 3 with the laptops domains. The two text fields `seq1` and `seq2` are the aspects and the review. The label is the sentiment. It's used in arXiv:1802.09913.

**Examples**


Review id:LPT1 (Laptop)

S1:The So called laptop Runs to Slow and I hate it! →
{LAPTOP#OPERATION_PERFORMANCE, negative}, {LAPTOP#GENERAL, negative}

S2:Do not buy it! → {LAPTOP#GENERAL, negative}

S3:It is the worst laptop ever. → {LAPTOP#GENERAL, negative}


**Basic stats:**

+ \# items = 3,710
+ \# labels = 3
    - 0: negative
    - 1: neutral
    - 2: positive

**Basic Unit**: sentence

**Split**: train : dev : test = 2618 : 291 : 801

**bibtex**
```
@article{pavlopoulos2014aspect,
  title={Aspect based sentiment analysis},
  author={Pavlopoulos, Ioannis},
  journal={Athens University of Economics and Business},
  year={2014}
}

arXiv:1802.09913

```

[**Webpage**](http://alt.qcri.org/semeval2016/task5/)

[arXiv:1802.09913](https://arxiv.org/abs/1802.09913)
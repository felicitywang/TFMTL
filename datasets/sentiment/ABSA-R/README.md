**ABSA-R -- Aspect-based Sentiment Analysis - Restaurants*

**Summary**

Aspect-based sentiment analysis is the task of identifying whether an aspect, i.e. a particular property of an item is associated with a positive, negative, or neutral sentiment.


This dataset is a sub-set of the SemEval 2016 Task 5 Subtask 1 Slot 3 with the restaurants domains. The two text fields `seq1` and `seq2` are the aspects and the tweet. The label is the sentiment. It's used in arXiv:1802.09913.

**Examples**

Review id:RST1 (Restaurant)

S1:I was very disappointed with this restaurant. →
{RESTAURANT#GENERAL, “restaurant”, negative, from="34" to="44"}

S2:I’ve asked a cart attendant for a lotus leaf wrapped rice and she replied back rice and just walked away. →{SERVICE#GENERAL, “cart attendant”, negative, from="12" to="26"}

S3:I had to ask her three times before she finally came back with the dish I’ve requested. →
{SERVICE#GENERAL, “NULL”, negative}

**Basic stats:**

+ \# items = 3,366
+ \# labels = 3
    - 0: negative
    - 1: neutral
    - 2: positive

**Basic Unit**: sentence

**Split**: train : dev : test = 2256 : 251 : 859

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
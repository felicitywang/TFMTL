**Stance Detection dataset for FNC-1**

**Summary**

The goal of fake news detection in the context of the Fake News Challenge 2 is to estimate whether the body of a news article agrees, disagrees, discusses, or is unrelated towards a headline. This dataset is from the first stage of the Fake News Challenge(FNC-1). The train-dev split is given by and used in arXiv:1802.09913.

**Basic stats:**

+ \# items = 75,385
+ \# labels = 4
    - 0: agree: the body text agrees with the headline
    - 1: disagree: the body text disagrees with the headline
    - 2: discuss: the body text discusses the same topic as the headline, but does not take a position
    - 3: unrelated: the body text discusses a different topic than the headline

**Basic Unit**: paragraph

**Split**: train : dev : test = 39,741 : 10,231 : 25,413

**bibtex**

```
arXiv:1802.09913
```

[FakeNewsChallenge.org](http://fakenewschallenge.org)

[arXiv:1802.09913](https://arxiv.org/abs/1802.09913)
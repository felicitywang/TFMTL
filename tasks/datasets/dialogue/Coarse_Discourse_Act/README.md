**Coarse Discourse Act**

**Summary**

This dataset consists of movie reviews from amazon. The data span a period of more than 10 years, including all ~8 million reviews up to October 2012. Reviews include product and user information, ratings, and a plaintext review. We also have reviews from all other Amazon categories.

**Basic stats:**
+ \# threads = 9,473
+ \# comments = 116,347
+ \# labels = 9 (I've removed 'other' in data.json)
    - 0: agreement
    - 1: announcement
    - 2: answer
    - 3: appreciation
    - 4: disagreement
    - 5: elaboration
    - 6: humor
    - 7: negativereaction
    - 8: question


**Basic Unit**: paragraph

** Note **:
    - title: only the initial post would have a title
    - text: title + content of this comment
    - parent_text: content of the parent of this comment

**bibtex**
```
@inproceedings{coarsediscourse, title={Characterizing Online Discussion Using Coarse Discourse Sequences}, author={Zhang, Amy X. and Culbertson, Bryan and Paritosh, Praveen}, booktitle={Proceedings of the 11th International AAAI Conference on Weblogs and Social Media}, series={ICWSM '17}, year={2017}, location = {Montreal, Canada} }
```

[**Webpage**](https://github.com/google-research-datasets/coarse-discourse)
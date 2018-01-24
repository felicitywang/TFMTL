**Coarse Discourse Act**

**Summary**

This dataset consists of movie reviews from amazon. The data span a period of more than 10 years, including all ~8 million reviews up to October 2012. Reviews include product and user information, ratings, and a plaintext review. We also have reviews from all other Amazon categories.

**Basic stats:**
+ \# posts = 101,498
+ \# labels = 9
    - 0: agreement
    - 1: announcement
    - 2: answer
    - 3: appreciation
    - 4: disagreement
    - 5: elaboration
    - 6: humor
    - 7: negativereaction
    - 8: question

*Text Fields*
    - title
    - body
    - parent_title
    - parent_body

**Basic Unit**: paragraph

*Split*: 10-fold split, keeping the posts of the same thread in the same split

** Note **:
    - title: only the initial post would have a title
    - text: title + content of this comment
    - parent_title: title of the parent post of this post
    - parent_content: content of the parent post of this post

**bibtex**
```
@inproceedings{coarsediscourse, title={Characterizing Online Discussion Using Coarse Discourse Sequences}, author={Zhang, Amy X. and Culbertson, Bryan and Paritosh, Praveen}, booktitle={Proceedings of the 11th International AAAI Conference on Weblogs and Social Media}, series={ICWSM '17}, year={2017}, location = {Montreal, Canada} }
```

[**Webpage**](https://github.com/google-research-datasets/coarse-discourse)

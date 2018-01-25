**Daily Dialog Emotion**

**Summary**
DailyDialog is created with human-written dialogs of various topics from English-learning websites and manually labeled communication intention and emotion information. This is the dailog intention part.


**Basic stats:**
+ \# sentences = 102,979 （13,118 dialogues)
+ \# labels = 7
    - 0: no emotion
    - 1: anger
    - 2: disgust
    - 3: fear
    - 4: happiness
    - 5: sadness
    - 6: surprise



**Basic Unit**: sentence

**Split: the author suggests a train:valid:test = 11,118 : 1000 : 1000 based on dialogue_id **
**Note: one such random split is provided but without 'topic' label, and the author suggests a random split by yourself**

**bibtex**
```
@InProceedings{li2017dailydialog,
                           author = {Li, Yanran and Su, Hui and Shen, Xiaoyu and Li, Wenjie and Cao, Ziqiang and Niu, Shuzi},
                           title = {DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset},
                           booktitle = {Proceedings of The 8th International Joint Conference on Natural Language Processing (IJCNLP 2017)},
                           year = {2017}
}
```

[**Webpage**](http://yanran.li/dailydialog)




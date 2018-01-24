**Daily Dialog Emotion**

**Summary**
DailyDialog is created with human-written dialogs of various topics from English-learning websites and manually labeled communication intention and emotion information. This is the dailog intention part.


**Basic stats:**
+ \# sentences = 102,979 （13,118 dialogues)
+ \# labels = 4
    - 1: inform(all statements and questions by which the speaker is providing information)
    - 2: question(when the speaker wants to know something and seeks for some information)
    - 3: directive(contains dialog acts like request, instruct, suggest and accept/reject offer)
    - 4: commissive(accept/reject request or suggestion and offer)

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




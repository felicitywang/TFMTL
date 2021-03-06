# Datasets

Datasets of different domains for text classification.

## Overview

| type      | name                            | #items     | #labels | unit      | summary                                              |
| --------- | ------------------------------- | ---------- | ------- | --------- | ---------------------------------------------------- |
| sentiment | SSTb                            | 11,855     | 5       | sentence  | Rotten Tomatoes movie reviews                        |
| sentiment | LMRD                            | 50,000     | 2       | document  | IMDB movie reviews                                   |
| sentiment | IMDb                            | 600,000    | 2       | paragraph | IMDb movie reviews                                   |
| sentiment | RTC                             | 43,800     | 2       | sentence  | Rotten Tomatoes critic movie reviews                 |
| sentiment | RTU                             | 739,903    | 2       | paragraph | Rotten Tomatoes user movie reviews                   |
| sentiment | SUBJ                            | 10,000     | 2       | sentence  | Rotten Tomatoes and IMDB movie reviews               |
| sentiment | Amazon_Movie_Reviews_Categories | 7,911,684  | 5       | document  | Amazon movie reviews                                 |
| sentiment | Amazon_Review_Full              | 3,650,000  | 5       | document  | Amazon product reviews                               |
| sentiment | Amazon_Review_Polarity          | 4,000,000  | 2       | document  | Amazon product review                                |
| sentiment | Yelp_Review_Full                | 700,000    | 5       | paragraph | Yelp reviews                                         |
| sentiment | Yelp_Review_Polarity            | 598,000    | 2       | paragraph | Yelp reviews                                         |
| sentiment | Twitter_Sentiment_Corpus        | 5,513      | 4       | paragraph | Sentiment tweets on 4 topics                         |
| sentiment | Yelp_Challenge                  | 4,736,897  | 5       | paragraph | Yelp review                                          |
| sentiment | Sentiment140                    | 1,600,498  | 3       | paragraph | Sentiment tweets on different brands/products/topics |
| sentiment | Target                          | 6,248      | 3       | sentence  | Sentiment tweets towards some entity                 |
| politics  | FGPS                            | 766        | 5       | sentence  | Political propositions                               |
| politics  | POLT                            | 318,761    | 2       | paragraph | Political tweets                                     |
| topic     | AG_News                         | 127,600    | 4       | document  | AG's news                                            |
| topic     | DBPedia                         | 63,000     | 14      | paragraph | DBPedia                                              |
| topic     | Sogou_News                      | 510,000    | 5       | document  | Sogou News(Chinese)                                  |
| topic     | Yahoo_Answers                   | 1,460,000  | 10      | document  | Yahoo! Answers                                       |
| dialogue  | Coarse_Discourse_Act            | 101,498    | 9       | paragraph | Reddit threads                                       |
| dialogue  | DailyDialog_Act                 | 102,979    | 4       | paragraph | English learning materials                           |
| emotion   | DailyDialog_Emotion             | 102,979    | 7       | paragraph | English learning materials                           |
| emotion   | Crowdflower_Emotion_in_Text     | 40,000     | 13      | paragraph | Emotional tweets                                     |
| emotion   | Emoti_Tweets                    | 38,900,000 | 8       | paragraph | Tweets with emotional hashtags, emoticons and emoji  |
| emotion   | NLP-dataset                     | 416,809    | 6       | paragraph | Unknown source                                       |
| emotion   | NRC_TEC                         | 21,051     | 6       | paragraph | Tweets self-labeled with hashtag annotations         |
| emotion   | Twitter_18                      | 2,524      | 18      | sentence  | Emotional tweets                                     |


## File structure
```
- datasets/
    - dataset_type/
        - dataset_name/
            - README.md
                - summary
                - basic stats
                - # items
                - # labels(number of labels and what each one means)
                - basic unit
                - citation(bibtex)
                - webpage link
            - format_data.py : python script which converts the original data into json format
            - text_field_names: names of all the text fields if not 'text'
            - label_field_name: name of the label field if not 'label'
```


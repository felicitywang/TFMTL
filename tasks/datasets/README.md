
# Overview

type|name|#items|#labels|unit|summary
---|---|---|---|---|---
sentiment|SSTb|11,855|5|sentence|Rotten Tomatoes movie reviews
sentiment|IMDB|50,000|2|document|IMDB movie reviews
sentiment|Amazon_Movie_Reviews_Categories|7,911,684|5|document|Amazon movie reviews
sentiment|Amazon_Review_Full|3,650,000|5|document|Amazon product reviews
sentiment|Amazon_Review_Polarity|4,000,000|2|document|Amazon product review
sentiment|Yelp_Review_Full|700,000|5|paragraph|Yelp reviews
sentiment|Yelp_Review_Polarity|598,000|2|paragraph|Yelp reviews
sentiment|Twitter_Sentiment_Corpus|5,513|4|paragraph|Sentiment tweets on 4 topics
sentiment|Yelp_Challenge|4,736,897|5|paragraph|Yelp review
sentiment|Sentiment140|1,600,498|3|paragraph|Sentiment tweets on different brands/products/topics
topic|AG_News|127,600|4|document|AG's news
topic|DBPedia|63,000|14|paragraph|DBPedia
topic|Sogou_News|510,000|5|document|Sogou News(Chinese)
topic|Yahoo_Answers|1,460,000|10|document|Yahoo! Answers
dialogue|Coarse_Discourse_Act|101,498|9|paragraph|Reddit threads
dialogue|DailyDialog_Act|102,979|4|paragraph|English learning materials
emotion|DailyDialog_Emotion|102,979|7|paragraph|English learning materials
emotion|Crowdflower_Emotion_in_Text|40,000|13|paragraph|Emotional tweets
emotion|Emoti_Tweets|38,900,000|8|paragraph|Tweets with emotional hashtags, emoticons and emoji
emotion|NLP-dataset|416,809|6|paragraph|Unknown source
emotion|NRC_TEC|21,051|6|paragraph|Tweets self-labeled with hashtag annotations
emotion|Twitter_18|2,524|18|sentence|Emotional tweets


### file structure
```
datasets/
dataset_type/
e.g.
sentiment/ : sentiment analysis
emotion/ : emotion detection
dialogue/ : dialogue discourse act
topic/ : topic

dataset_name/
e.g. IMDB, SSTb, etc.
README.md
summary
basic stats
# items
# labels(number of labels and what each one means)
basic unit
citation(bibtex)
webpage link
format_data.py : python script which converts the original data into json format
original/ : all the original data
data.json.gz : all the original data
json block containing text fields, label field and metadata
the 'label' field(or perhaps with other name) are all integers
index.json.gz : indices of train/test/(valid) if given
text_field_names: names of all the text fields if not 'text'
label_field_name: name of the label field if not 'label'
```
### path
`/export/b02/fwang/mlvae/tasks/datasets/`
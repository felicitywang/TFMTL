type|dataset|accuracy|min_freq
---|---|---|---
sentiment|SSTb|40.7240%|1
sentiment|IMDB|89.0160%|50
sentiment|Twitter_Sentiment_Corpus|77.3438%|0
emotion|NRC_TEC|61.1586%|2
emotion|NLP-dataste|90.4633%|50
emotion|Twitter_18|22.1344%|2
emotion|Crowdflower_Emotion_in_Text|34.8250%|10
domain|MATERIAL|57.1429%|0

### hyperparameters:
- learning rate: 0.00005
- dropout rate:0.5
- batch size: 32
- seed: 42
- max_num_epoch: 20(with early stopping)
- layers: [100, 100]
- encoding: bag of words
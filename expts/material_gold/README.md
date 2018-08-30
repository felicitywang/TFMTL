# README


# data

## original data

There's some document-level data in two languages: Swahili(1A) and Tagalog(1B) and two types: text and speech text. Each document is labeled with some domains and has human English translations. Altogether, there're 609 documents in 1A and 621 documents in 1B.


|domain|language|num_pos| num_all|
|-|-|-|-|
| GOV | 1A, 1B | 217 + 209 = 426 | 1,230 |
| LIF | 1A, 1B | 200 + 204 = 404| 1,230 |
| BUS | 1A | 206 | 609 |
| LAW | 1A | 208 | 609 |
| HEA | 1B | 193 | 621 |
| MIL | 1B | 192 | 621 |


## synthetic data

For each domain, some in-domain and out-of-domain keywords(for annotators to better distinguish domains). Top 1000 results by searching such keywords in Wikipedia pages with Lucene are saved as positive examples. 1000 randomly selected Wikipedia pages are saved as negative examples(same for each domain).

## experiment data

For each domain, a STL model is trained with such splits:
train: synthetic data, 100 or 1000 positive examples, 1000 negative examples
dev:  half of the gold data
test: the other half of the gold data
(the positive and negative gold examples are evenly distributed to dev and test splits)

|dataset|train pos | train neg |dev pos| dev neg |test pos | test neg |
|-|-|-|-|-|-|-|
| GOV_100 | 100 | 1000 | 213 | 402 | 213 | 402 |
| LIF_100 | 100 | 1000 | 202|  413 | 202 | 413 |
| BUS_100 | 100 | 1000 | 103 | 201 | 103 | 202 |
| LAW_100 | 100 | 1000 | 104 | 200 | 104 | 199 |
| HEA_100 | 100 | 1000 |  96 | 214 | 97 | 214 |
| MIL_100 | 100 | 1000 |  96| 214 | 96 | 215 |
| GOV_1000 | 1000 | 1000 | 213 | 402 | 213 | 402 |
| LIF_1000 | 1000 | 1000 | 202 | 413 | 202 | 413 |
| BUS_1000 | 1000 | 1000 | 103 | 201 | 103 | 202 |
| LAW_1000 | 1000 | 1000 | 104 | 200 | 104 | 199 |
| HEA_1000 | 1000 | 1000 |  96 | 214 | 97 | 214 |
| MIL_1000 | 1000 | 1000 |  96| 214 | 96 | 215 |


# extractors

paragram_phrase(simply averaging of all the word embeddings in a sequence) + a non-linear(relu) layer


# word embedding

- no pretrianed word embedding

- glove.6B.300d: Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 300d)

# hyperparameters

## preprocessing


- max_document_length: 400/1000/-1
- max_vocab_size: -1
- min_frequency: 1
- max_frequency: -1
- random_seed: 42
- subsample_ratio: 1
- padding: false
- preproc: true(remove html tags etc)
- tokenizer: tweet_tokenizer(keep punctuations)


## discriminative driver

- num_train_epochs 30
- optimizer rmsprop
- lr0 0.001
- patience 3
- shared_mlp_layers 1
- shared_hidden_dims 100
- private_mlp_layers 1
- private_hidden_dims 100
- input_keep_prob 1.0
- output_keep_prob 0.5
- l2_weight 0
- tuning_metric Acc
- seed 42
- metrics Acc F1_PosNeg_Macro Precision_Macro Recall_Macro Confusion_Matrix
- reporting_metric Ac

## paragram encoder
- embed_dim: 300
- reducer: reduce_max_over_time
- apply_activate: true
- activation_fn: tf.nn.relu


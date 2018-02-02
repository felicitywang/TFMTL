# README

## Code

The newest code is in branch `multitask`.

`dataset.py`: reads `data.json.gz`(and possibly `index.json.gz` if standard splits are given) and writes TFRecord files

`mlp.py`: the MLP model

`cnn.py`: the CNN model

`test_single.py`: test the MLP/CNN model on the single dataset


## Datasets

See `../datasets/README.md`

1. All the json datasets are in `/export/b02/fwang/mlvae/tasks/datasets/` on the grid. Only `data.json.gz` (and `index.json.gz`, `text_field_names`, `label_field_name` if there are any) are needed.

2. You can also check ../../expts/sentiment_1/README.md and generate json datasets from the original files. The two datasets share the vocabulary in the experiment.

## Run

- an example testing MLP with bag of words:

```
python3 test_single.py --data_dir=../datasets/sentiment/SSTb/ --min_freq=1 --encoding=bow --model=mlp --padding=True
```

- an example testing CNN:

```
python3 test_single.py --data_dir=../datasets/sentiment/LMRD/ --min_freq=50 --encoding=word_id --model=cnn --padding=False
```

- Check `test_single.py` and `dataset.py` for all the arguments.

- I use `../scripts/qsub_mlp` to tune hyperparameters on the grid. Probably needs fixing since I added some more arguments and changed some paths.


## Results

There are some results on some smaller datasets in `../report/mlp.md`. Some datasets don't converge with bow MLP and I haven't figured out why.


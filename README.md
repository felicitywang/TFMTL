# TFMTL: A TensorFlow Framework for Multi-Task Learning

TFMTL is a full-pipeline framework for multi-task learning text classification tasks developed in TensorFlow. You can download formatted text datasets, preprocess datasets, configure the embedding/encoding/FFN architectures by running a few scripts and modify the configurations in some JSON files. You can also easily add your own modules following the standard input/output. 

Apart from the easy configuration of model architectures, multi-task learning(MTL) allows you to run multiple tasks together, with each you can configure a different architecture or choose the share parts of the models for better transfer. Pre-training and fine-tuning are also supported for better transfer.  

Because of the easy usage, flexible configuration for all text tasks, and the support for multi-task learning, this codebase is especially useful for running non-contextual baselines for text classification tasks; As it's developed in TensorFlow(when there's only TensorFlow 1.x), it can also serve as a good reference/learning tutorial for beginners/developers in TensorFlow for text tasks.  

## Installation

### Python 3 and Virtual Environment
This codebase is written in Python 3. We recommend using conda for your virtual environment. 

In case you're not familiar with conda, below is an example script to create a new virtual environment with conda:
```bash
conda create --name tfmtl
conda activate tfmtl
conda install pip
```

### Requirements

The requirements are listed in `requirements.txt`. To install, run:
```
pip install -U pip
pip install -r requirements.txt
```

## Development

To install either

```python setup.py develop```

or 

```pip install mtl -e```

Either allows you use the `mtl` package in an editable mdoe. 


## File Structure

- `datasets/`: collection of different datasets of text classification tasks with summaries, dataset statistics and bibtex info; for each dataset we also provide a script to download the dataset and convert them to a standard format in JSON. 
- `expts/`:
  - `scripts/`: scripts to configure the run the model end-to-end
  - `example/`: example code to run experiments with a detailed tutorial
    - `experiment_name/`: setup, configuration, running scripts, etc. for a particular experiment
- `mtl/`: source code of the mtl package
    - `extractors`: modeuls for encoding layers(no-op DAN, CNN, LSTM, etc., each with further configuration support)
    - `embedders`: modeuls for the embedding layers(no-op(for BOW), embedding, pre-trained word embedding, etc.)
    - `optim`: optimizers
    - `util`: helper functions
- `pretrained_word_embeddings/`: folder to save the pre-trained word embedding files with easy downloading scripts
- `tests/`: test files


## Running Experiments

`expts/README.md` provides a detailed tutorial on how to modify the configurations and run experiments.


## Embedders, Extractors, and Encoders
`ENCODER_README.md` provides a detailed documentation of how the model is organized and how to develop your own model. 


## Citation
Our paper `Bag-of-Words Transfer: Non-Contextual Techniques for Multi-Task Learning` is accepted by [DeepLo-2019](https://sites.google.com/view/deeplo19/home)(Deep Learning for Low-Resource NLP Workshop, EMNLP 2019). It's focused on non-contextual ways for multi-task learning and uses this codebase. If you use this codebase, please cite our paper and add a footnote of this repo.


```
@inproceedings{ebner-etal-2019-bag,
    title = "Bag-of-Words Transfer: Non-Contextual Techniques for Multi-Task Learning",
    author = "Ebner, Seth  and
      Wang, Felicity  and
      Van Durme, Benjamin",
    booktitle = "Proceedings of the 2nd Workshop on Deep Learning Approaches for Low-Resource NLP (DeepLo 2019)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-6105",
    pages = "40--46",
    abstract = "Many architectures for multi-task learning (MTL) have been proposed to take advantage of transfer among tasks, often involving complex models and training procedures. In this paper, we ask if the sentence-level representations learned in previous approaches provide significant benefit beyond that provided by simply improving word-based representations. To investigate this question, we consider three techniques that ignore sequence information: a syntactically-oblivious pooling encoder, pre-trained non-contextual word embeddings, and unigram generative regularization. Compared to a state-of-the-art MTL approach to textual inference, the simple techniques we use yield similar performance on a universe of task combinations while reducing training time and model size.",
}
```

## License
2-Clause BSD License

This research is based upon work supported in part by the Intelligence Advanced Research Projects Activity (IARPA), (contract FA8650-17-C-9115). The views and conclusions herein are those of the authors and should not be interpreted as necessarily representing official policies, ex-pressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copy-right annotation therein.

## Contact
GitHub issues are the best way for questions. Author emails can be found in our paper. 

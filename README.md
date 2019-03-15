# A TensorFlow Framework for Multi-Task Learning

[![pipeline status](https://gitlab.hltcoe.jhu.edu/vandurme/tfmtl/badges/master/pipeline.svg)](https://gitlab.hltcoe.jhu.edu/vandurme/tfmtl/commits/master)

## Attribution

If code in this repository contributes to work to be submitted for
publication, please consult the author(s) of the relevant code to
agree on appropriate attribution. Depending on the usage, one of these
may be appropriate:

* No special attribution necessary. This will most often be the case;
  for instance, for low-level infrastructure or simple baselines.
* A mention in the **Acknowledgements** section of the eventual
  publication.
* Co-author on the publication. This may be appropriate if you are
  relying on large portions of code that you did not write yourself.

Please refrain from sharing code in this repository beyond immediate
collaborators, as parts of it may be related to work that is still under
submission or will soon be submitted for publication.

## Requirements

The requirements are listed in `requirements.txt`. To install, run:

```
pip install -U pip
pip install -r requirements.txt
```

## Development

* Set up the package using `python setup.py develop`
* Implement tests by subclassing `tf.test.TestCase`. Put tests in
  `tests` and affix `"test_*.py"` or `"*_test.py"`.
* Follow the TensorFlow style guidelines.
* You must submit work via pull (merge) requests; do not push
  to `master` directly.
* Make sure `run_checks.sh` completes before merging into
  `origin/master`. You may need to run the following first:

  ``` bash
  pip install flake8
  pip install -U setuptools
  ```

* Use `six` to maintain backwards compatibility with Python 2.7.

## File Structure

- `datasets/`: collection of different datasets of text classification tasks with summaries, dataset statistics and bibtex info

- `expts/`:
  - `scripts/`: scripts to write TFRecord data and to run the model
  - `example/`: example code to run experiments
  - `experiment_name/`: setup, configuration, running scripts, etc. for a particular experiment
  - ...
  <!-- - TODO example folders -->
- `mtl/`: main codebase

- `pretrained_word_embeddings/`: folder to save the pre-trained word embedding files

- `tests/`: test files


## Running Experiments

See `expts/README.md` for a detailed experiment pipeline.


## Embedders, Extractors, and Encoders
See `ENCODER_README.md`.

<!-- TODO detailed lists of each encoder and corresponding arguments? perhaps in another place?
## Embedders, Extractors, and Encoders

One focus of this repository is a collection of off-the-shelf functions
for transforming inputs into feature representations. These transformations
are referred to as "encoders", and they consist of two steps: an embedding
step that turns input token IDs into token embeddings, and an extraction step
that turns the embeddings into feature representations. Together, the embedder
and extractor constitute an encoder.

An architecture is a specification of an encoder for each dataset in an experiment.
Users specify encoder architectures in a JSON configuration file. An architecture
is specified with a name (key) whose value is a dictionary with the following fields:

* `embedders_tied`:
  * `true` if all datasets should be embedded with the same function (i.e., shared parameters),
  * `false` otherwise
* `extractors_tied`:
  * `true` if all datasets should have features extracted with the same function (i.e., shared parameters),
  * `false` otherwise
* Name of dataset A
  * `embed_fn`: a string specifying an embedding function
  * `embed_kwargs`: a dictionary specifying `argument: value` pairs (arguments are strings) for the embedding function
  * `extract_fn`: a string specifying an extraction function
  * `extract_kwargs`: a dictionary specifying `argument: value` pairs (arguments are strings) for the extraction function
* Name of dataset B...

`embed_fn`, `embed_kwargs`, `extract_fn`, and `extract_kwargs` must be fully specified,
even if `embedders_tied` or `extractors_tied` is `true`. If `embedders_tied` is `true`
for an architecture, then `embed_fn` and `embed_kwargs` must have identical values for
all datasets specified in the given architecture (similarly if `extractors_tied` is `true`).

`true` is the JSON equivalent of Python's `True` value.

`false` is the JSON equivalent of Python's `False` value.

`null` is the JSON equivalent of Python's `None` value.

An example configuration file can be found at `tfmtl/tests/encoders.json`.

Embedding functions and extraction functions can be found in `tfmtl/mtl/embedders`
and `tfmtl/mtl/extractors`, respectively.

The architecture to use in an experiment is given by the `--architecture` flag,
and the file containing the architecture(s) is given by the `--encoder_config_file`
flag. Multiple architectures can be placed in the same configuration file. -->


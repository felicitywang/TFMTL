# A TensorFlow Framework for Multi-Task Learning

[![pipeline status](https://gitlab.hltcoe.jhu.edu/vandurme/tfmtl/badges/ci/pipeline.svg)](https://gitlab.hltcoe.jhu.edu/vandurme/tfmtl/commits/ci)

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
collaborators, as parts of it may related to work that is still under
submission or will soon be submitted for publication.

## Requirements

The requirements are listed in requirements.txt. To install, run:

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

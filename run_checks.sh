#! /usr/bin/env bash

set -e

flake8 mtl/decoders
flake8 mtl/encoders
flake8 mtl/models
flake8 mtl/util
flake8 mtl/mlvae
flake8 mtl/hparams.py
flake8 mtl/io.py
flake8 mtl/layers

pytest

echo "All checks pass!"

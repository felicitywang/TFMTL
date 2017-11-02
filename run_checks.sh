#! /usr/bin/env bash

set -e

flake8 mlvae
python setup.py test

echo "All checks pass!"

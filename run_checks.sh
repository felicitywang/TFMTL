#! /usr/bin/env bash

set -e

flake8
python setup.py test

echo "All checks pass!"

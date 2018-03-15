#! /usr/bin/env bash

set -e

python setup.py test
flake8

echo "All checks pass!"

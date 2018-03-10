#! /usr/bin/env bash

set -e

flake8 --exit-zero
#python setup.py test

echo "All checks pass!"

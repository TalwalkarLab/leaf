#!/usr/bin/env bash

NAME="synthetic"

cd ../utils

python3 stats.py --name $NAME

cd ../$NAME
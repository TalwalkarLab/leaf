#!/usr/bin/env bash

NAME="sent140"

cd ../utils

python3 stats.py --name $NAME

cd ../$NAME
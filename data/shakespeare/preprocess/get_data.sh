#!/usr/bin/env bash

cd ../data/raw_data

wget --adjust-extension http://www.gutenberg.org/files/100/100-0.txt
mv 100-0.txt raw_data.txt

cd ../../preprocess
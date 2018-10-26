#!/usr/bin/env bash

cd ../data/raw_data

if [ ! -f trainingandtestdata.zip ]; then
    wget --no-check-certificate http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
fi

unzip trainingandtestdata.zip

mv training.1600000.processed.noemoticon.csv training.csv
mv testdata.manual.2009.06.14.csv test.csv

rm trainingandtestdata.zip

cd ../../preprocess
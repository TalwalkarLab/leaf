#!/usr/bin/env bash

# download data and convert to .json format

if [ ! -f "data/all_data/data.json" ]; then
	echo "Please run the main.py script to generate the initial data."
	exit 1
fi


NAME="synthetic" # name of the dataset, equivalent to directory name
 

cd ../utils

./preprocess.sh --name $NAME $@

cd ../$NAME
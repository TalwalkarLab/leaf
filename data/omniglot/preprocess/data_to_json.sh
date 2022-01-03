#!/usr/bin/env bash

# Setup data and raw_data directories, if they don't already exist
if [ ! -d "../data/raw_data" ]; then
    mkdir -p ../data/raw_data
fi

# Check and download data if needed
./get_data.sh

if [ ! "$(ls -A ../data/all_data)" ]; then
    mkdir -p ../data/all_data
    echo "------------------------------"
    echo "converting data to .json format"
    python3 data_to_json.py
    echo "finished converting data to .json format"
fi

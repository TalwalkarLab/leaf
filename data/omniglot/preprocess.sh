#!/usr/bin/env bash
set -x

# download data and convert to .json format
if [ ! -d "data/all_data" ] || [ ! "$(ls -A data/all_data)" ]; then
    pushd preprocess
    ./data_to_json.sh
    popd
fi

NAME="omniglot" # name of the dataset, equivalent to directory name

cd ../utils

./preprocess.sh --name $NAME $@

cd ../$NAME

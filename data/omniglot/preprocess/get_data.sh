#!/usr/bin/env bash

DOWNLOAD_URL="https://raw.githubusercontent.com/brendenlake/omniglot/master/python/"
declare -a data_folders=( "images_background" "images_evaluation" )

pushd "../data/raw_data"
    echo "------------------------------"
    for data_folder in "${data_folders[@]}"; do
        if [ ! -d "${data_folder}" ]; then
            echo "Downloading ${data_folder}"
            wget --no-check-certificate "${DOWNLOAD_URL}/${data_folder}.zip"
            unzip "${data_folder}.zip"
            rm ${data_folder}.zip
        else
            echo "Found Omniglot image directory ${data_folder}"
        fi
    done
popd

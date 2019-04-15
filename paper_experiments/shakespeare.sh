#!/usr/bin/env bash

output_dir="${1:-'./baseline'}"

split_seed=""
sampling_seed=""

declare -a local_epochs=( "20" "1" )


###################### Functions ###################################

function move_data() {
    path="$1"
    suffix="$2"
    
    pushd models/metrics
        mv sys_metrics.csv "${path}/sys_metrics_${suffix}.csv"
        mv stat_metrics.csv "${path}/stat_metrics_${suffix}.csv"
    popd

    cp -rf data/shakespeare/meta "${path}"
    mv "${path}/meta" "${path}/meta_${suffix}"
}

function run_experiment() {
    num_epochs="${1}"
    pushd models/
        python -u main.py -dataset shakespeare -model stacked_lstm --seed 0 --num-rounds 80 \
                  --clients-per-round 10 --num_epochs ${num_epochs} -lr 0.8
    popd
    move_data ${output_dir} "shakespeare_c_10_rnd_80_e_${num_epochs}"
}


##################### Script #################################
pushd ../

# Check that data and models are available
if [ ! -d 'data/' -o ! -d 'models/' ]; then
    echo "Couldn't find data/ and/or models/ directories - please run this script from the root of the LEAF repo"
fi

# Check that output directory is available
mkdir -p ${output_dir}
output_dir=`realpath ${output_dir}`
echo "Storing results in directory ${output_dir} (please invoke this script as: ${0} <dirname> to change)"

# If data unavailable, execute pre-processing script
if [ ! -d 'data/shakespeare/data/train' ]; then
    if [ ! -f 'data/shakespeare/preprocess.sh' ]; then
        echo "Couldn't find data/ and/or models/ directories " \
             "- please obtain scripts from GitHub repo: https://github.com/TalwalkarLab/leaf"
        exit 1
    fi

    echo "Couldn't find Shakespeare data - " \
         "running data preprocessing script"
    pushd data/shakespeare/
        rm -rf meta/ data/all_data data/test data/train data/rem_user_data data/intermediate
        ./preprocess.sh -s niid --sf 0.05 -k 64 -tf 0.9 -t sample
    popd
fi

# Run local epoch experiment
for num_epoch in "${local_epochs[@]}"; do
    echo "Running Shakespeare experiment with ${num_epoch} local epochs"
    run_experiment "${num_epoch}"
done

popd

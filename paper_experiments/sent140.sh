#!/usr/bin/env bash

OUTPUT_DIR=${1:-"./baseline"}

# Each value is (k, seed) pair
declare -a k_vals=( "100 1549774894" "30 1549775083" "10 1549775860" "3 1549780473" )

###################### Functions ###################################

function get_k_data() {
	keep_clients="${1?Please provide value of keep_clients}"
	split_seed="${2}"

	pushd data/sent140/
		rm -rf meta/ data/
		./preprocess.sh --sf 0.5 -k ${keep_clients} -s niid -t sample --spltseed ${split_seed}
	popd
}

function move_data() {
	path="$1"
	suffix="$2"
	
	pushd models/metrics
		mv sys_metrics.csv "${path}/sys_metrics_${suffix}.csv"
		mv stat_metrics.csv "${path}/stat_metrics_${suffix}.csv"
	popd

	cp -r data/sent140/meta "${path}"
	mv "${path}/meta" "${path}/meta_${suffix}"
}

function run_k() {
	k="$1"
	get_k_data "$k"
	pushd models
		python main.py -dataset 'sent140' -model 'stacked_lstm' --num-rounds 10 --clients-per-round 2
	popd
	move_data ${OUTPUT_DIR} "k_${k}"
}

###################### Script ########################################
pushd ../

if [ ! -d "data/" -o ! -d "models/" ]; then
	echo "Couldn't find data/  and/or models/ directories - please run this script from the root of the LEAF repo"
fi

# Check that preprocessing scripts are available
if [ ! -d 'data/sent140/preprocess' ]; then
	echo "Please obtain preprocessing scripts from LEAF GitHub repo: https://github.com/TalwalkarLab/leaf"
	exit 1
fi

mkdir -p ${OUTPUT_DIR}
OUTPUT_DIR=`realpath ${OUTPUT_DIR}`
echo "Writing output files to ${OUTPUT_DIR}"

# Check that GloVe embeddings are available; else, download them
pushd models/sent140
	if [ ! -f glove.6B.300d.txt ]; then
		./get_embs.sh
	fi
popd

for val_pair in "${k_vals[@]}"; do
	k_val=`echo ${val_pair} | cut -d' ' -f1`
	seed=`echo ${val_pair} | cut -d' ' -f2`
	run_k "${k_val}" "${seed}"
	echo "Completed k=${k_val}"
done

popd

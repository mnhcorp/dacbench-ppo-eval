#!/bin/bash

# check if $1 and $2 are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ];
then
    echo "Please provide the experiment name, difficulty (easy, medium or hard) and number of runs"
    echo "Example: ./run.sh ppo1 easy 3"
    exit 1
fi

exp=$1_$2
mode=$2
runs=$3
conf="configs/train_conf_ppo_$mode.yml"

mkdir -p experiments/$exp

python ../scripts/run_ppo.py -o experiments/$exp -s $conf -n $runs
# for each run copy the config file and analyse the results, loop starts from 0
for i in $(seq 0 $((runs-1)))
do
    echo "Generating plots for run $i.."
    tgt=experiments/$exp/run_$i
    cp $conf $tgt/config.yml
    python analyse_results.py $tgt
done


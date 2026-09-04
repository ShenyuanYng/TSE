#!/usr/bin/env bash

# (1) run on slurm...
cmd_num_nodes=$1
cmd_num_workers=$2
cmd_dependency="$3"

mem=10G # this one is usually enough for single-GPU jobs
mem=20G
if [ "$cmd_dependency" != "" ]; then
    export cuda_cmd="slurm.pl --gpu $cmd_num_nodes --num-threads $cmd_num_workers --config tools/slurm.conf --path ./path.sh --mem $mem --dependency $cmd_dependency"
else
    export cuda_cmd="slurm.pl --gpu $cmd_num_nodes --num-threads $cmd_num_workers --config tools/slurm.conf --path ./path.sh --mem $mem"
fi

# (2) run locally...
#export sdr_cmd="run.pl"
#export cuda_cmd="run.pl"

#!/bin/bash
# Evaluate the pass@1 result of HumanEval

filename=$1
flag=$2

pushd "/root/Humaneval"
./run_humaneval.sh "$filename"
popd

echo "=================================================="

python eval_speed.py $filename $flag

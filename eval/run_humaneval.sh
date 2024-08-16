#!/bin/bash

flag=$1

# Greedy
if [ "$flag" = "greedy" ]; then
    echo "Greedy"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    USE_LADE=1 LOAD_LADE=1 \
    python humaneval.py \
        --model_name_or_path "meta-llama/Llama-2-70b-hf" \
        --input_file "Humaneval_Solution.jsonl" \
        --output_file "./Output/jsonls/greedy-70b.jsonl" \
        --level 4 \
        --window 5 \
        --guess 5 \
        > ./Output/logs/greedy-70b.log
# Sample
elif [ "$flag" = "sample" ]; then
    echo "Sample"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    USE_LADE=1 LOAD_LADE=1 \
    python humaneval.py \
        --model_name_or_path "meta-llama/Llama-2-70b-hf" \
        --input_file "Humaneval_Solution.jsonl" \
        --output_file "./Output/jsonls/sample-70b.jsonl" \
        --sample \
        --topp 0.95 \
        --temperature 0.8 \
        --level 4 \
        --window 5 \
        --guess 5 \
        > ./Output/logs/sample-70b.log
# No LADE
elif [ "$flag" = "no" ]; then
    echo "No LADE"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python humaneval.py \
        --model_name_or_path "meta-llama/Llama-2-70b-hf" \
        --input_file "Humaneval_Solution.jsonl" \
        --output_file "./Output/jsonls/70b-sample.jsonl" \
        --sample \
        --topp 0.95 \
        --temperature 0.1
else
    echo "Please enter the correct flag!"
fi

#!/bin/bash
# Run humaneval.py with different flags

flag=$1

# Greedy
if [ "$flag" = "greedy" ]; then
    echo "Greedy"
    USE_LADE=1 LOAD_LADE=1 \
    python humaneval.py \
        --model_name_or_path "meta-llama/Llama-2-7b" \
        --input_file "Humaneval_Solution.jsonl" \
        --output_file "./Output/Policy/greedy-7b-W4-fifo.jsonl" \
        --level 5 \
        --window 4 \
        --guess 4 \
        --device "gpu:3"
# Sample
elif [ "$flag" = "sample" ]; then
    echo "Sample"
    USE_LADE=1 LOAD_LADE=1 \
    python humaneval.py \
        --model_name_or_path "meta-llama/Llama-2-7b" \
        --input_file "Humaneval_Solution.jsonl" \
        --output_file "./Output/sample-7b-W8.jsonl" \
        --sample \
        --topp 0.95 \
        --temperature 0.8 \
        --level 5 \
        --window 8 \
        --guess 8 \
        --device "gpu:7"
# No LADE
elif [ "$flag" = "no" ]; then
    echo "No LADE"
    python humaneval.py \
        --model_name_or_path "meta-llama/Llama-2-7b" \
        --input_file "Humaneval_Solution.jsonl" \
        --output_file "./Output/no-greedy-7b.jsonl" \
        --device "gpu:5"
else
    echo "Please enter the correct flag!"
fi


# GUESS_SET_SIZE=-1
# USE_LADE=1 LOAD_LADE=1 \
# python humaneval.py \
#     --model_name_or_path "meta-llama/Llama-2-7b" \
#     --input_file "Humaneval_Solution.jsonl" \
#     --output_file "./Output/Policy/greedy-7b-5.jsonl" \
#     --level 5 \
#     --window 5 \
#     --device "gpu:2"

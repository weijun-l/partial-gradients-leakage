#!/bin/bash
array=( $@ )
len=${#array[@]}
last_args=${array[@]:6:$len}

python3 dpsgd_model_utility.py --dataset $1 --batch_size $2 --bert_path $3 --num_epochs $4 --noise_multiplier $5 --clipping_bound $6  $last_args
#!/bin/bash
array=( $@ )
len=${#array[@]}
last_args=${array[@]:5:$len}

python3 partial_attack.py --dataset $1 --split test --loss cos --n_inputs 10 -b $2 --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 --lr_decay 0.89 --bert_path $3 --n_steps 2000 --grad_type $4 --attack_layer $5 $last_args
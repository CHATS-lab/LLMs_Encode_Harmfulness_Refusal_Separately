#!/bin/bash


python -u ../src/inference.py \
	--input "../data/" \
	--output_file_name "output" \
	--load_ckpt 0 \
	--peft_pth_ckpt "" \
	--model "qwen" \
	--left 0 \
	--right 100 \
	--use_jb 0 \
	--max_len 128 \
	--model_size "7b" \
	--record_prob_max_pos 0 \
	--use_template 1 \
	--do_not_use_last_inst_tok 1 \
	--use_inversion 0 \
	--inversion_prompt_idx 6


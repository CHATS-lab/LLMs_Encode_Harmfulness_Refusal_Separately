#!/bin/bash

python -u ../src/extract_hidden.py \
	--harmful_pth "$1" \
	--harmless_pth "$2" \
	--use_persuade_harmful $3 \
	--use_sys_harmful 0 \
	--use_persuade_harmless 0 \
	--output_pth "$4" \
	--output_pth_harmful "$5" \
	--output_pth_harmless "$6" \
	--left $7 \
	--right $8 \
	--random_sample_harmful $9 \
	--model "${10}" \
	--extract_hidden_inst_token 1 \
	--extract_harmful_token_only 0 \
	--batch_size 1 \
	--ret_whole_seq 0 \
	--model_size "7b" \
	--load_ckpt 0 \
	--mode_dir "${11}"
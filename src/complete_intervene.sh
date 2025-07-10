python -u ../src/intervention.py \
	--test_data_pth "data/example_test_harmless.json" \
	--output_pth "output/inversion.json" \
	--intervention_vector "../data/pt/" \
  --reverse_intervention 1 \
  --intervene_context_only 1 \
	--arg_key_prompt 'instruction' \
	--use_persuade_test 0 \
  --model "qwen" \
  --left 0 \
  --right 50 \
  --layer_s 0 \
  --layer_e 28 \
  --coeff_select 2 \
  --max_token_generate 100 \
  --max_decode_step_while_intervene 1 \
	--model_size "7b" \
  --use_inversion 1 \
	--inversion_prompt_idx 1



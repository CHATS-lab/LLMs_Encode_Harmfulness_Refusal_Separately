#!/bin/sh

# Define models and keys as space-separated strings

model="qwen"
mode_dir="hf"

harmful_pth="data/"
use_persuade_harmful=0 #1: use jailbroken version of harmful data
harmless_pth="data/"

# Create output paths with model and key information
output="out_pt/dir-hf-hl.pt"
output_harmful="out_pt/harmful.pt"
output_harmless="out_pt/harmless.pt"

sh get_diff_mean.sh "$harmful_pth" "$harmless_pth" "$use_persuade_harmful" "$output" "$output_harmful" "$output_harmless" 0 50 0 "$model" "$mode_dir"


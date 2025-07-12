

For intervention experiments,  run complete_intervene.sh  

Key parameters:
- `--intervention_vector`: Path to steering vectors
- `--reverse_intervention`: Whether to reverse intervention (1/0)
- `--coeff_select`: Coefficient selection for intervention strength
- `--use_inversion`: Whether to do reply inversion (1/0)
- `--intervene_context_only`: Whether to only apply intervention to tokens before the inversion question. set as 1 when running reply inversion task with harmfulness directions

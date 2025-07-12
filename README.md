<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> LLMs Encode Harmfulness and Refusal Separately </h1>


**Content warning**: This repository contains text that is offensive, harmful, or otherwise inappropriate in nature.

This repository contains the official implementation for the paper **"LLMs Encode Harmfulness and Refusal Separately"**. Our research reveals that large language models (LLMs) encode harmfulness and refusal as distinct concepts in their latent representations.

- [paper](https://arxiv.org/abs/XXXX.XXXXX)


### Key Findings

- **Separate Encoding**: Hidden states at the last token of instruction tokens (`t_inst`) encode harmfulness, while the last token of post-instruction tokens (`t_post-inst`) encodes refusal behavior
- **Causal Evidence**: Steering along the harmfulness direction changes the model's internal perception of harmfulness, while the refusal direction only affects surface-level refusal characteristics
- **Jailbreak Analysis**: Some jailbreak methods work by suppressing refusal signals without altering the model's internal harmfulness judgment
- **Latent Guard**: Internal harmfulness representations can serve as safeguards for detecting unsafe inputs

##  Project Structure

```
src/
├── Core Scripts
│   ├── extract_hidden.py          # Extract hidden states from LLMs
│   ├── intervention.py            # Controlled text generation with interventions
│   ├── inference.py               # Model inference on datasets
│   ├── eval.py                    # Evaluation utilities
│   ├── utils.py                   # Helper functions
│   ├── template_inversion.py      # Templates for reply inversion task
│   ├── run_llama_guard.py         # LlamaGuard evaluation
│   └── classifier.ipynb           # Jupyter notebook for latent guard
|
└── Shell Scripts
|   ├── complete_intervene.sh      # Full intervention pipeline
|   ├── run_diff_mean.sh           # Hidden state extraction 
│   └── run_inference.sh           # Inference pipeline
│
└── run/
    pt files #example extracted directions and hidden states
```

## Experiments
### Hidden State Analysis

Our analysis focuses on two key token positions:
- **`t_inst`**: Last token of the user's instruction (encodes harmfulness)
- **`t_post-inst`**: Last token of the entire input prompt (encodes refusal behavior)

```bash
sh run_diff_mean.sh
```
This will reproduce hidden states for two specified clusters (e.g., harmful prompts and harmless prompts) and according difference 
By default, we extract last 2 instruction tokens + all the special post-instruction tokens. 

### Intervention Experiments

Perform controlled interventions to modify model behavior:

```bash
# Run intervention with specific parameters
sh complete_intervene.sh
```

Key parameters:
- `--intervention_vector`: Path to steering vectors
- `--reverse_intervention`: Whether to reverse the steering vector (1/0)
- `--use_inversion`: Whether to do reply inversion task (1/0)



### Latent Guard Implementation

One of our contribution is the **Latent Guard** - an intrinsic safeguard that uses the model's own internal harmfulness representations.
Implementations are in `classifier.ipynb`.

#### Compare with Baselines

```bash
# Run LlamaGuard 3 with Ollama
python run_llama_guard.py --input data/test_prompts.json --output results/llamaguard.txt
```


## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zhao2025decoupling,
  title={Decoupling Harmfulness from Refusal in LLMs},
  author={Zhao, Jiachen and Huang, Jing and Wu, Zhengxuan and Bau, David and Shi, Weiyan},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```


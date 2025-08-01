<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> LLMs Encode Harmfulness and Refusal Separately </h1>


**Content warning**: This repository contains text that is offensive, harmful, or otherwise inappropriate in nature.

This repository contains the official implementation for the paper **"LLMs Encode Harmfulness and Refusal Separately"**. Our research reveals that large language models (LLMs) encode harmfulness and refusal as distinct concepts in their latent representations.

- [Paper](https://arxiv.org/abs/2507.11878)
- [Website](https://chats-lab.github.io/LLMs_Encode_Harmfulness_Refusal_Separately/)
- [Blog](https://www.lesswrong.com/posts/gzNe2Grj2KksvzHWM/llms-encode-harmfulness-and-refusal-separately)


### Key Findings

- **Separate Encoding**: Hidden states at the last token of instruction tokens (`t_inst`) encode harmfulness, while the last token of post-instruction tokens (`t_post-inst`) encodes refusal behavior
- **Causal Evidence**: Steering along the harmfulness direction changes the model's internal perception of harmfulness, while the refusal direction only affects surface-level refusal characteristics
- **Jailbreak Analysis**: Some jailbreak methods work by suppressing refusal signals without reversing the model's internal harmfulness judgment
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
This will reproduce hidden states for two specified clusters (e.g., harmful prompts and harmless prompts) and the according direction from one cluster to the other.

### Intervention Experiments

Perform controlled interventions to modify model behavior:

```bash
# Run intervention with specific parameters
sh complete_intervene.sh
```

Key parameters:
- `--intervention_vector`: Path to the steering vectors
- `--reverse_intervention`: Whether to reverse the steering vector (1/0)
- `--use_inversion`: Whether to do reply inversion task (1/0)


### Latent Guard Implementation

One of our contributions is the **Latent Guard** - an intrinsic safeguard that uses the model's own internal harmfulness representations.
Implementations are in `classifier.ipynb`.

#### Compare with Baselines

```bash
# Run LlamaGuard 3 with Ollama
python run_llama_guard.py --input data/{test_prompts_name}.json --output results/llamaguard.txt
```


## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{zhao2025llmsencodeharmfulnessrefusal,
      title={LLMs Encode Harmfulness and Refusal Separately}, 
      author={Jiachen Zhao and Jing Huang and Zhengxuan Wu and David Bau and Weiyan Shi},
      year={2025},
      eprint={2507.11878},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.11878}, 
}
```


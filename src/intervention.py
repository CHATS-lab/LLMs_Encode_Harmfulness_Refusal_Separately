"""
Text Generation with Intervention Vectors

This module provides functionality for performing controlled text generation using
intervention vectors that can modify the behavior of language models during inference.
It supports various intervention strategies including activation addition, attention
masking, and layer-specific modifications.

Key Features:
- Activation vector intervention at specific layers
- Configurable intervention positions and coefficients
- Probability and attention recording capabilities
"""

import json
import torch
import os
import copy
import argparse
from typing import List, Tuple, Callable, Optional, Union
from torch import Tensor
from tqdm import tqdm
from utils import read_row, formatInp_llama_persuasion, ret_top_attn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import contextlib
import functools
import numpy as np
from template_inversion import inversion_prompts_choice
import random
from path import *

# Constants
DECODING_STEP = 3
MODEL = 'llama'
COUNT_ADD=0

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """Context manager for adding and removing hooks from modules."""
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))

        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_activation_addition_input_pre_hook(
    vector: Tensor, 
    coeff: Tensor,
    cache: Optional[List] = None,
    record: int = 0,
    intervene_all: bool = True,
    positions: List[int] = None
):
    """Hook function to add activation vectors to model inputs."""
    if cache is None:
        cache = []
    if positions is None:
        positions = [-1]

    def hook_fn(module, input):
        nonlocal vector, cache
        global COUNT_ADD
        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input
        
        vector = vector.to(activation)
        
        if DECODING_STEP == -1 or COUNT_ADD < DECODING_STEP:  # when equal to -1, till end of generation
            assert coeff != 0
            if intervene_all:
                activation += coeff * vector
            else:
                for pos in positions:
                    activation[:, pos, :] += coeff * vector
            COUNT_ADD+=1
        if record:
            cache.append(activation)

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def complete_with_intervention(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    instructions: List[dict], 
    tokenize_instructions_fn: Callable,
    intervene_layers: List[int], 
    batch_size: int = 32, 
    intervention_vector_ori: Optional[Tensor] = None,
    args: Optional[dict] = None
) -> List[dict]:
    """
    Complete text generation with intervention vectors.
    
    Args:
        model: The language model to use for generation
        tokenizer: The tokenizer for the model
        instructions: List of instruction dictionaries
        tokenize_instructions_fn: Function to tokenize instructions
        intervene_layers: List of layer indices to intervene on
        batch_size: Batch size for processing
        intervention_vector_ori: Original intervention vectors
        args: Configuration arguments dictionary
        
    Returns:
        List of completion dictionaries containing generated text and metadata
    """
    coeff = torch.tensor(float(args['add_coef_intervene']))
    generation_config = GenerationConfig(
        max_new_tokens=args['max_token_generate'], 
        do_sample=False
    )
    generation_config.pad_token_id = tokenizer.pad_token_id
    
    n_layers = 28 if MODEL == 'qwen' else 32
    ret = []
    cache = [[] for _ in range(n_layers)]
    global COUNT_ADD
    for i in tqdm(range(0, len(instructions), batch_size)):
        raw_inp = instructions[i][args['arg_key_prompt']]  # assume batch size always 1

        if intervention_vector_ori.shape[0] > 1:
            intervention_vector = intervention_vector_ori[i].squeeze()
        else:
            intervention_vector = intervention_vector_ori.squeeze()

        print('intervention vector shape', intervention_vector_ori.shape)
        print('intervention vector shape reshape', intervention_vector.shape)
        
        tokenized_instructions = tokenize_instructions_fn(instructions[i:i + batch_size])
        seq_len = tokenized_instructions.input_ids.shape[-1]
        intervene_position = list(range(seq_len))

        if args['intervene_context_only']:
            if MODEL == 'qwen':
                special_token = inversion_prompts_choice[args['inversion_prompt_idx']] + "<|im_end|>\n<|im_start|>assistant"
            elif MODEL == 'llama3':
                special_token = inversion_prompts_choice[args['inversion_prompt_idx']] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            elif MODEL == 'llama2':
                special_token = inversion_prompts_choice[args['inversion_prompt_idx']] + "[/INST]"
            special_token_len = len(tokenizer.tokenize(special_token))
            print('special token length', special_token_len)
            intervene_position = list(range(seq_len - special_token_len))
            
        print(50 * "#")
        print('intervene with steering vectors')
        print('intervene position', intervene_position) 
        
        fwd_pre_hooks = [
            (model.model.layers[intervene_layer],
             get_activation_addition_input_pre_hook(
                 vector=intervention_vector[intervene_layer, :], 
                 coeff=coeff,
                 intervene_all=args['intervene_all'],
                 positions=intervene_position
             )) for intervene_layer in intervene_layers
        ]

        completions = []

        print('index', i)
        print('length of instructions', len(instructions))
        assert i < len(instructions)

        if not args['intervene_all']:
            for id in intervene_position:
                print('intervening token', tokenizer.convert_ids_to_tokens(tokenized_instructions.input_ids[0])[id])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            generation_toks = model.generate(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                output_attentions=True,
                use_cache=not args['mask_attn_token']
            )
                
            generated_scores = generation_toks.scores
            prob = []
            tokens = []
            
            if args['record_probs']:
                for ii, score in enumerate(generated_scores):
                    if ii < 10:
                        probability = torch.softmax(score[0], dim=-1)
                        all_probs = probability.detach().cpu().numpy().tolist()
                        token_id = torch.argmax(probability).item()
                        prob_id = probability[token_id].item()
                        token = tokenizer.decode(token_id)
                        prob.append(all_probs)
                        tokens.append(token)
                    else:
                        break
                        
            print('length of input token', tokenized_instructions.input_ids.shape[-1])
            generation_toks = generation_toks.sequences[:, tokenized_instructions.input_ids.shape[-1]:]  # remove input tokens
            print('length of generated token', generation_toks.shape[-1])

            for generation_idx, generation in enumerate(generation_toks):
                completions.append({
                    'prompt': instructions[i + generation_idx],
                    'response': tokenizer.decode(generation, skip_special_tokens=True).strip(),
                    'tokens': tokens,
                    'probs': prob,
                })
            COUNT_ADD=0
        ret.extend(completions)
            
        if len(cache[0]) > 0:
            print(len(cache), len(cache[0]))
            flat_list = [torch.stack(inner_list) for inner_list in cache]
            result = torch.stack(flat_list).squeeze()
            print(result.shape)  # layer, #example, pos, hidden dim
            mean_activations = result.mean(dim=1)
            #torch.save(result, args['output_pth'].replace('.json', '_attn.pt'))
            #torch.save(mean_activations, args['output_pth'].replace('.json', '_mean_attn.pt'))
            
    return ret


def main():
    """
    Run the full text generation pipeline with intervention vectors.
    
    This function sets up the model, loads intervention vectors, and performs
    controlled text generation with various intervention strategies.
    """
    parser = argparse.ArgumentParser(description="Text generation with intervention vectors")
    
    # Model configuration
    parser.add_argument("--model", default='llama', type=str, help="Model type")
    parser.add_argument("--model_size", default='7b', type=str, help="Model size")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    
    # Data paths
    parser.add_argument("--test_data_pth", default='data/medcq.json', type=str, help='Test data path')
    parser.add_argument("--output_pth", default='data/medcq.json', type=str, help="Output file path")
    parser.add_argument("--intervention_vector", default=None, type=str, help='Intervention vector path')
    
    # Processing parameters
    parser.add_argument('--left', default=0, type=int, help='Left index')
    parser.add_argument('--right', default=10, type=int, help='Right index')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--mode', default='complete', type=str, help='Mode')
    
    # Intervention parameters
    parser.add_argument('--add_coef_intervene', default=1., type=float, help='Intervention coefficient')
    parser.add_argument('--layer_s', default=0, type=int, help='Start layer')
    parser.add_argument('--layer_e', default=32, type=int, help='End layer')
    parser.add_argument('--intervention_vector_layer', default=-1, type=int, help='Intervention vector at a layer')
    parser.add_argument('--reverse_intervention', default=0, type=int, help='Reverse intervention')
    parser.add_argument('--intervene_all', default=0, type=int, help='Intervene all tokens')
    parser.add_argument('--intervene_context_only', default=0, type=int, help='Intervene context only before the inversion prompt, used for reply inversion task only')
    
    # Generation parameters
    parser.add_argument('--max_token_generate', default=400, type=int, help='Max tokens to generate')
    parser.add_argument('--max_decode_step_while_intervene', default=1, type=int, help='Max decode steps while intervening')

    # Recording parameters
    parser.add_argument('--record_probs', default=0, type=int, help='Record probabilities')
 
    # Model-specific parameters
    parser.add_argument('--arg_key_prompt', default='bad_q', type=str, help='Argument key for prompt')
    parser.add_argument('--remove_inst_inp', default=0, type=int, help='Remove instruction input')
    parser.add_argument('--use_inversion', default=0, type=int, help='Use inversion reply task')
    parser.add_argument('--inversion_prompt_idx', default=0, type=int, help='Inversion prompt index')
    # Used parameters
    parser.add_argument('--use_jailbreak_test', default=0, type=int, help='Use the jailbreak version of inputs at test')
    parser.add_argument('--coeff_select', default=1, type=float, help='Coefficient select')
    parser.add_argument('--traverse_single_layer_intervention', default=0, type=int, help='Traverse single layer intervention')
    parser.add_argument('--load_ckpt', default=0, type=int, help='Load checkpoint')
    parser.add_argument('--peft_pth_ckpt', default='output/attn.json', type=str, help='PEFT checkpoint path')
    parser.add_argument('--positions', default='-1', type=str, help='Positions')
    
    args = parser.parse_args()
    params = vars(args)
    
    global MODEL, DECODING_STEP
    DECODING_STEP = params['max_decode_step_while_intervene']
    MODEL = params['model']
    
    # Parse list arguments
    params['intervene_pos'] = list(map(int, params['intervene_pos'].split()))
    params['positions'] = list(map(int, params['positions'].split()))
    

    # Model loading with error handling
    try:
        if MODEL == 'llama':
            llama_2_model_path = "NousResearch/Llama-2-7b-chat-hf"

            model = AutoModelForCausalLM.from_pretrained(
                llama_2_model_path,
                cache_dir='models/llama',
                torch_dtype=torch.float16,
                device_map="cuda",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                llama_2_model_path,
                cache_dir='models/llama',
            )
        elif MODEL == 'llama3':
            local_pth = 'models/llama3-hf/LLM-Research/Meta-Llama-3-8B-Instruct'
            if not os.path.exists(local_pth):
                local_pth = 'models/llama3'
            model = AutoModelForCausalLM.from_pretrained(
                local_pth,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(local_pth)
            tokenizer.pad_token = tokenizer.eos_token 
        elif MODEL == 'qwen':
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2-7B-Instruct", 
                trust_remote_code=True, 
                cache_dir='models/qwen'
            )
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2-7B-Instruct", 
                device_map="auto", 
                trust_remote_code=True,
                cache_dir='models/qwen'
            )
        else:
            raise ValueError(f"Unsupported model type: {MODEL}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if params['load_ckpt']:
        print('loading ckpt adapter')
        try:
            model.load_adapter(params['peft_pth_ckpt'])
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return

    if params['mode'] == 'complete':
        try:
            test_data = read_row(params['test_data_pth'])[params['left']:params['right']]
        except Exception as e:
            print(f"Error loading test data: {e}")
            return
            
        try:
            intervention_vector = torch.load(params['intervention_vector'])
        except Exception as e:
            print(f"Error loading intervention vector: {e}")
            return

        candidate_coeff = [params['coeff_select']] #if params['coeff_select'] else [1] 
        
        for coeff in candidate_coeff:
            params['add_coef_intervene'] = coeff
            print('coefficient for intervention', coeff)
            
            layer_list = list(range(13, 32)) if params['traverse_single_layer_intervention'] else [0]
            
            for out_i in layer_list:
                for i in range(params['layer_s'], params['layer_e']):
                    print('##################')
                    print('intervention layer', i)                        
                    use_persuade = params['use_jailbreak_test']
                    if isinstance(intervention_vector, np.ndarray):
                        intervention_vector = torch.tensor(intervention_vector)
                    if len(intervention_vector.shape) == 2:
                        intervention_vector = intervention_vector.unsqueeze(0)
                        
                    print('intervention vector shape', intervention_vector.shape)

                    if params['intervention_vector_layer'] >= 0:
                        intervention_vector_local = intervention_vector[:, params['intervention_vector_layer'], :].to(model.device)
                    elif params['traverse_single_layer_intervention']:
                        intervention_vector_local = intervention_vector[:, out_i, :].to(model.device)
                    else:
                        if intervention_vector.shape[0] != 1:
                            intervention_vector_local = intervention_vector[params['left']:params['right']].to(model.device)
                        else:
                            intervention_vector_local = intervention_vector.to(model.device)

                    if params['reverse_intervention']:
                        intervention_vector_local = -intervention_vector_local

                    def tokenize_instructions_fn(instructions, use_persuade=use_persuade):
                        inps = [
                            formatInp_llama_persuasion(
                                i, use_persuade, model=MODEL, 
                                use_inversion=params['use_inversion'],
                                inversion_prompt_idx=params['inversion_prompt_idx']
                            ) for i in instructions
                        ]
                        return tokenizer(inps, padding=True, return_tensors="pt")

                    try:
                        completions = complete_with_intervention(
                            model, tokenizer, test_data, tokenize_instructions_fn,
                            [i], batch_size=params['batch_size'], 
                            intervention_vector_ori=intervention_vector_local,
                            args=params,
                        )
                            
                        # Save results
                        output_file = params['output_pth'].replace('.json', f'-intervene{i}.json')
                        with open(output_file, 'w') as f:
                            for completion in completions:
                                json.dump(completion, f)
                                f.write("\n")
                                
                    except Exception as e:
                        print(f"Error during intervention for layer {i}: {e}")
                        continue


if __name__ == "__main__":
    main()
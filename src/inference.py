
import os
import torch
import argparse
import math
import json
import logging
from typing import Dict, List, Tuple, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

from utils import read_row, formatInp_llama_persuasion

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_device(0)
else:
    raise RuntimeError("CUDA is required for this script")

def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    generation_config: GenerationConfig,
    args: Dict[str, Any]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Evaluate a model on a given prompt.
    
    Args:
        model: The language model to evaluate
        tokenizer: The tokenizer for the model
        prompt: Input prompt text
        generation_config: Configuration for text generation
        args: Additional arguments dictionary
        
    Returns:
        Tuple of (generated_text, probability_scores)
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    
    max_new_tokens = args['max_len']
    
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1
    )
    
    generated_scores = generation_output.scores
    logger.debug(f'Max tokens: {max_new_tokens}, Input length: {len(input_ids[0])}')
    logger.debug(f'Generated scores: {len(generated_scores)}, Generated IDs: {len(generation_output.sequences[0])}')
    
    # Extract probability scores
    probabilities = []
    for i, score_tensor in enumerate(generated_scores):
        if i >= args['record_prob_max_pos']:
            break
            
        probability = torch.softmax(score_tensor[0], dim=-1)
        all_probs = probability.detach().cpu().numpy().tolist()
        token_id = torch.argmin(probability).item()
        token = tokenizer.decode(token_id)
        probabilities.append({'prob': all_probs, 'token': token})
    
    # Decode the generated text
    output = tokenizer.decode(generation_output.sequences[0])
    return output, probabilities


def extract_model_output(
    full_output: str,
    model_type: str,
    input_prompt: str = ""
) -> str:
    """
    Extract the model's response from the full generated text.
    
    Args:
        full_output: Complete generated text
        model_type: Type of model used
        input_prompt: Original input prompt (for llama3)
        
    Returns:
        Extracted model response
    """
    if model_type == 'llama' or model_type == 'mistral':
        return full_output.split('[/INST]')[-1].replace('</s>', '').strip()
    elif model_type == 'vicuna':
        return full_output.split('ASSISTANT:')[-1].replace('</s>', '').strip()
    elif model_type == 'qwen':
        return full_output.split('assistant')[-1].replace('</s>', '').strip()
    elif model_type == 'llama3':
        return full_output.split(input_prompt)[-1].strip()
    else:
        return full_output.strip()


def infer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_data: List[Dict[str, Any]],
    args: Dict[str, Any]
) -> None:
    """
    Run inference on evaluation data.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        eval_data: List of evaluation data points
        args: Configuration arguments
    """
    model.eval()
    model.config.use_cache = True
    
    # Remove existing output file
    if os.path.exists(args['output_file_name']):
        os.remove(args['output_file_name'])
    
    # Configure generation strategy
    if args['do_sample_decode']:
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=args['temperature'],
            top_p=args['top_p'],
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
    else:
        generation_config = GenerationConfig(
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    logger.info('Starting inference...')

    for i in tqdm(range(len(eval_data)), desc="Processing samples"):
        data_point = eval_data[i]
        input_prompt = formatInp_llama_persuasion(
            data_point,
            use_persuade=args['use_jb'],
            use_adv=args['use_adv_suffix'],
            use_ss=args['use_sys_prompt'],
            model=args['model'],
            use_template=args['use_template'],
            do_not_use_last_inst_tok=args['do_not_use_last_inst_tok'],
            use_inversion=args['use_inversion'],
            inversion_prompt_idx=args['inversion_prompt_idx']
        )
        
        output, probabilities = evaluate(model, tokenizer, input_prompt, generation_config, args)
        # Process and save results
        with open(args['output_file_name'], 'a') as f:
            if args['use_jb']:
                response = extract_model_output(output, args['model'])
                eval_data[i] = {'prompt': eval_data[i], 'response': response}
            else:
                eval_data[i]['ori_output'] = extract_model_output(output, args['model'], input_prompt)
            
            eval_data[i]['probs'] = probabilities
            json.dump(eval_data[i], f)
            f.write('\n')
        
        logger.debug(f'Output: {output[:100]}...')


def load_model_and_tokenizer(model_type: str, model_size: str, load_checkpoint: bool = False, checkpoint_path: str = "") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the specified model and tokenizer.
    
    Args:
        model_type: Type of model ('llama', 'llama3', 'qwen')
        model_size: Size of model ('7b', '13b', '14b')
        load_checkpoint: Whether to load a checkpoint
        checkpoint_path: Path to checkpoint if loading
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        if model_type == 'llama':
            #specify your model path here
            model_path = "NousResearch/Llama-2-7b-chat-hf" if model_size == '7b' else "Llama-2-13b-chat-hf/"
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                #cache_dir='models/llama',
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                #cache_dir='models/llama',
            )
            
        elif model_type == 'llama3':
            model_path = "unsloth/Meta-Llama-3.1-8B-Instruct"
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                #cache_dir='models/llama3',
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                #cache_dir='models/llama3',
            )
            
        elif model_type == 'qwen':
            if model_size == '7b':
                model_path = "Qwen/Qwen2-7B-Instruct"
                #cache_dir = 'models/qwen'
            elif model_size == '14b':
                model_path = "Qwen/Qwen2.5-14B-Instruct"
                #cache_dir = 'models/qwen-14b'
            else:
                raise ValueError(f"Unsupported model size: {model_size}")
                
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                #cache_dir=cache_dir
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                #cache_dir=cache_dir
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load checkpoint if specified
        if load_checkpoint and checkpoint_path:
            model.load_adapter(checkpoint_path)
            
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model {model_type}: {e}")
        raise


def main():
    """Main function to run the inference script."""
    parser = argparse.ArgumentParser(description="Language model inference script")
    
    # Model configuration
    parser.add_argument("--model", default='llama3', type=str, help="Model type (llama, llama3, qwen)")
    parser.add_argument("--model_size", default='7b', type=str, help="Model size (7b, 13b, 14b)")
    parser.add_argument("--peft_pth_ckpt", default='', type=str, help="Path to PEFT checkpoint")
    parser.add_argument("--load_ckpt", default=0, type=int, help="Whether to load checkpoint")
    
    # Data configuration
    parser.add_argument("--input", default='data/medcq.json', type=str, help="Input file path")
    parser.add_argument("--output_file_name", default='data/medcq.json', type=str, help="Output file path")
    parser.add_argument('--left', default=0, type=int, help='Left index for data slicing')
    parser.add_argument('--right', default=10, type=int, help='Right index for data slicing')
    
    # Generation configuration
    parser.add_argument("--max_len", default=4096, type=int, help="Maximum generation length")
    parser.add_argument("--temperature", default=0, type=float, help="Generation temperature")
    parser.add_argument("--top_p", default=0, type=float, help="Top-p sampling parameter")
    parser.add_argument("--do_sample_decode", default=0, type=int, help="Use sampling for decoding")
    parser.add_argument("--record_prob_max_pos", default=3, type=int, help="Max positions to record probabilities")
    
    # Prompt configuration
    parser.add_argument("--use_jb", default=0, type=int, help="Use jailbroken prompt")
    parser.add_argument("--use_adv_suffix", default=0, type=int, help="Use adversarial suffix")
    parser.add_argument("--use_sys_prompt", default=0, type=int, help="Use system prompt")
    parser.add_argument("--use_template", default=0, type=int, help="Use default prompting template")
    parser.add_argument("--do_not_use_last_inst_tok", default=0, type=int, help="Don't use last instruction token")
    parser.add_argument("--use_inversion", default=0, type=int, help="Use inversion")
    parser.add_argument("--inversion_prompt_idx", default=0, type=int, help="Inversion prompt index")
    
    
    args = parser.parse_args()
    params = vars(args)
    
    logger.info(f'Model: {params["model"]}')
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        params['model'],
        params['model_size'],
        params['load_ckpt'],
        params['peft_pth_ckpt']
    )
    
    # Load and preprocess data
    test_data = read_row(params['input'])
    
    # Validate indices
    if params['left'] < 0:
        params['left'] = 0
    if params['right'] > len(test_data):
        params['right'] = len(test_data)
    
    if params['left'] >= params['right']:
        raise ValueError("Left index must be less than right index")
    
    # Filter data
    test_data = [
        d for d in test_data[params['left']:params['right']]
        if 'sample_rounds' not in d or d['sample_rounds'] != 'Failed'
    ]
    
    # Run inference
    infer(model, tokenizer, test_data, params)


if __name__ == "__main__":
    main()

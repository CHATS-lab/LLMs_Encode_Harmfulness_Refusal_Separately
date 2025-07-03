import requests
import time
import argparse
from utils import read_row
MODEL_NAME = "llama-guard3:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,required=True, help="Path to the input file with prompts")
    parser.add_argument("--output", type=str, required=True, help="Path to the output file to save results")
    parser.add_argument("--use_persuade", type=int, default=0, help="Path to the input file with prompts")
    args = parser.parse_args()


    prompts=read_row(args.input)[:500]
    if 'prompt' in prompts[0]:
        prompts=[e['prompt'] for e in prompts]
    if 'bad_q' in prompts[0]:
        
        if 'ss_prompt' in prompts[0] and args.use_persuade:
            print('use ss_prompt')
            prompts=[e['ss_prompt'] for e in prompts]
        elif 'adv_suffix' in prompts[0] and not args.use_persuade:
            print('use adv_suffix')
            prompts=[e['bad_q']+e['adv_suffix'] for e in prompts]
        else:
            print('use bad_q')
            prompts=[e['bad_q'] for e in prompts]
    elif 'instruction' in prompts[0]:
        print('use instruction')
        prompts=[e['instruction'] for e in prompts]
    assert len(prompts)>0
    # Evaluate prompts using Ollama
    def evaluate(prompt):
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    # Run evaluation
    results = []
    unsafe_cnt=0
    for i, prompt in enumerate(prompts):
        try:
            output = evaluate(prompt)
            is_unsafe = "unsafe" in output.lower() 
            if is_unsafe:
                unsafe_cnt+=1
            print(f"[{i+1}/100] Prompt: {prompt}")
            print(f"Response: {output}")
            print("-" * 50)
            results.append((prompt, output))
            time.sleep(0.2)  # throttle slightly if needed
        except Exception as e:
            print(f"Error on prompt {i+1}: {e}")
            results.append((prompt, "ERROR"))

    print(f'unsafe count:{unsafe_cnt}')
    print(f'unsafe rate:{unsafe_cnt/len(prompts)*100}')
    # Save to a file
    with open(args.output, "w") as f:
        for prompt, response in results:
            f.write(f"Prompt: {prompt}\nResponse: {response}\n{'='*40}\n")
        f.write(f'unsafe count:{unsafe_cnt}\n')
        f.write(f'unsafe rate:{unsafe_cnt/len(prompts)*100}\n')
if __name__ == "__main__":
    main()

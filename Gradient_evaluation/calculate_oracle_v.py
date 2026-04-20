import argparse
import json
import os
import re
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============ Global runtime state ============
_model = None
_tok = None
_n_oracle = None
_batch_size = None
_do_sample = True
_temperature = 1.0
_top_p = 0.95
_max_new_tokens = 32

# ============ Reward function ============
_INT_PAT = re.compile(r"(-?\d+)")

def reward_01_from_text(text: str, gold: int) -> float:
    """Extract the final integer from text and compare it with gold."""
    m = _INT_PAT.findall(text)
    if not m:
        return 0.0
    try:
        pred = int(m[-1])
    except Exception:
        return 0.0
    return 1.0 if pred == gold else 0.0

# ============ Oracle V computation ============
@torch.no_grad()
def compute_oracle_v_single(example: dict) -> dict:
    """Compute Oracle V for a single example."""
    global _model, _tok, _n_oracle, _batch_size
    global _do_sample, _temperature, _top_p, _max_new_tokens
    
    prompt = example.get("prompt")
    gold = example.get("ground_truth")
    device = _model.device
    
    # Normalize prompt format to a plain text prompt.
    if isinstance(prompt, list):
        # Chat-style prompt.
        try:
            prompt_text = _tok.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in prompt])
    else:
        # Plain string prompt.
        prompt_text = prompt
    
    # Tokenize
    enc = _tok(prompt_text, return_tensors="pt").to(device)
    
    # Monte Carlo sampling.
    remaining = _n_oracle
    correct = 0
    
    while remaining > 0:
        bs = min(_batch_size, remaining)
        out = _model.generate(
            **enc,
            do_sample=_do_sample,
            temperature=_temperature,
            top_p=_top_p,
            max_new_tokens=_max_new_tokens,
            num_return_sequences=bs,
            pad_token_id=_tok.eos_token_id,
        )
        
        prompt_len = enc["input_ids"].shape[1]
        cont = out[:, prompt_len:]
        texts = _tok.batch_decode(cont, skip_special_tokens=True)
        
        for t in texts:
            correct += int(reward_01_from_text(t, gold))
        
        remaining -= bs
    
    V = correct / float(_n_oracle)
    
    # Attach computed statistics to the example.
    example["V_oracle"] = float(V)
    example["n_oracle"] = int(_n_oracle)
    example["correct_count"] = int(correct)
    
    return example

# ============ Entry point ============
def main():
    global _model, _tok, _n_oracle, _batch_size
    global _do_sample, _temperature, _top_p, _max_new_tokens
    
    parser = argparse.ArgumentParser(description="Compute Oracle V for each dataset example.")
    
    # Model arguments.
    parser.add_argument("--model", type=str, required=True, help="model name or local path")
    parser.add_argument("--revision", type=str, default="main", help="model revision")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    
    # Dataset arguments.
    parser.add_argument("--dataset", type=str, required=True, help="dataset ID or local path")
    parser.add_argument("--split", type=str, default="train", help="dataset split")
    
    # Computation arguments.
    parser.add_argument("--n_oracle", type=int, default=1000, help="number of samples per example")
    parser.add_argument("--batch_size", type=int, default=32, help="generation batch size")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    
    # Sampling arguments.
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    
    # Output arguments.
    parser.add_argument("--out_dir", type=str, default="oracle_v_results")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Configure dtype.
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # Load model and tokenizer.
    print(f"\nLoading model: {args.model}")
    _tok = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    if _tok.pad_token_id is None:
        _tok.pad_token = _tok.eos_token
    
    _model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    _model.to(args.device)
    _model.eval()
    print("Model loaded")
    
    # Set global runtime parameters.
    _n_oracle = args.n_oracle
    _batch_size = args.batch_size
    _do_sample = True
    _temperature = args.temperature
    _top_p = args.top_p
    _max_new_tokens = args.max_new_tokens
    
    torch.manual_seed(args.seed)
    
    # Load dataset.
    print(f"\nLoading dataset: {args.dataset}")
    from datasets import load_dataset
    dataset = load_dataset(args.dataset, split=args.split)
    print(f"Dataset loaded, total examples: {len(dataset)}")
    
    # Process full dataset.
    print("\nComputing Oracle V for all examples...")
    t_start = time.time()
    
    dataset_with_v = dataset.map(compute_oracle_v_single, desc="Compute Oracle V", batched=False)
    
    # Save results.
    model_name = args.model.split("/")[-1].replace("-", "_")
    ts = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(args.out_dir, f"oracle_v_n{args.n_oracle}_{model_name}_{ts}.jsonl")
    
    print("\nSaving results...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in dataset_with_v:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    elapsed = time.time() - t_start
    avg_time = elapsed / len(dataset)
    
    print("\nDone")
    print(f"  - examples: {len(dataset_with_v)}")
    print(f"  - elapsed: {elapsed / 60:.2f} minutes")
    print(f"  - avg: {avg_time:.2f} sec/example")
    print(f"  - output: {jsonl_path}")

if __name__ == "__main__":
    main()

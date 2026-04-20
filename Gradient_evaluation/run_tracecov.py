import argparse
import json
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm


from typing import List, Dict, Tuple, Optional
import numpy as np


# -----------------------------
# 0) reward: parse final integer and compare with gold
# -----------------------------
_INT_PAT = re.compile(r"(-?\d+)")

def reward_01_from_text(text: str, gold) -> float:
    # Extract answer
    matches = _INT_PAT.findall(text)
    if not matches:
        return 0.0
    
    try:
        pred = int(matches[-1])
    except (ValueError, IndexError):
        return 0.0
    
    # Convert ground_truth to integer (may be string format)
    try:
        gt_int = int(gold)
    except (ValueError, TypeError):
        return 0.0
    
    # Compare
    if pred == gt_int:
        return 1.0
    else:
        return 0.0

# -----------------------------
# 1) per-parameter Welford trace(cov)
# -----------------------------
class RunningTraceCov:
    """
    Tracks per-parameter (elementwise) covariance trace via Welford:
      trace(cov) = sum_j Var(z_j)
    where z_j are gradient components.
    """
    def __init__(self, params, device: str):
        self.params = list(params)
        self.device = device
        self.n = 0
        self.mean = []
        self.M2 = []
        for p in self.params:
            self.mean.append(torch.zeros_like(p, dtype=torch.float32, device=device))
            self.M2.append(torch.zeros_like(p, dtype=torch.float32, device=device))

    @torch.no_grad()
    def reset(self):
        self.n = 0
        for i in range(len(self.params)):
            self.mean[i].zero_()
            self.M2[i].zero_()

    @torch.no_grad()
    def update_from_grads(self):
        self.n += 1
        n = self.n
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            z = p.grad.detach().to(torch.float32)
            delta = z - self.mean[i]
            self.mean[i].add_(delta / n)
            delta2 = z - self.mean[i]
            self.M2[i].add_(delta * delta2)

    @torch.no_grad()
    def trace_cov(self) -> float:
        if self.n < 2:
            return 0.0
        denom = self.n - 1
        tr = 0.0
        for i in range(len(self.params)):
            tr += (self.M2[i] / denom).sum().item()
        return tr

# -----------------------------
# 2) data loading (JSONL or Parquet)
# -----------------------------
def load_oracle_table(path: str, limit: Optional[int] = None, seed: int = 0) -> List[Dict]:
    """
    Return list of dict items: {id, prompt, gold, V_oracle, ...}
    Supports .jsonl or .parquet
    """
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if limit is not None and len(rows) > limit:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(rows), size=limit, replace=False)
            rows = [rows[i] for i in idx]
        return rows

    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        if limit is not None and len(df) > limit:
            df = df.sample(n=limit, random_state=seed)
        return df.to_dict(orient="records")

    raise ValueError("oracle_table must be .jsonl or .parquet")

# -----------------------------
# 3) Batched sampling: generate N = M*K continuations in chunks
# -----------------------------

@torch.no_grad()
def sample_N(
    model, tok,
    messages,          # list[{"role":..., "content":...}]
    N: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    gen_bs: int,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[str]]:

    device = model.device

    prompt_text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tok(prompt_text, return_tensors="pt").to(device)

    prompt_ids = enc["input_ids"][0].detach().cpu()
    L = enc["input_ids"].shape[1]

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_id = tok.eos_token_id

    cont_ids_list: List[torch.Tensor] = []
    texts: List[str] = []

    remaining = int(N)
    while remaining > 0:
        bsz = min(remaining, int(gen_bs))

        out = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=bsz,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

        for k in range(bsz):
            cont = out[k, L:].detach().cpu()

            if eos_id is not None:
                eos_pos = (cont == eos_id).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    cont = cont[: int(eos_pos[0].item())]

            if pad_id is not None and pad_id != eos_id:
                pad_pos = (cont == pad_id).nonzero(as_tuple=False)
                if pad_pos.numel() > 0:
                    cont = cont[: int(pad_pos[0].item())]

            cont_ids_list.append(cont)
            texts.append(tok.decode(cont, skip_special_tokens=True))

        remaining -= bsz

    return prompt_ids, cont_ids_list, texts


def pad_and_stack(seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(seqs)
    Lmax = max(int(s.numel()) for s in seqs) if B > 0 else 0
    input_ids = torch.full((B, Lmax), pad_id, dtype=torch.long)
    attn = torch.zeros((B, Lmax), dtype=torch.long)
    for i, s in enumerate(seqs):
        L = int(s.numel())
        input_ids[i, :L] = s
        attn[i, :L] = 1
    return input_ids, attn

def batched_logprob_continuations(model, tok, prompt_ids: torch.Tensor, cont_ids_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute log pi(y_k | x) for each continuation y_k via teacher forcing in one forward pass.
    Returns lp: (K,) with grad graph.
    """
    device = model.device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    full = []
    Lp = int(prompt_ids.numel())
    for c in cont_ids_list:
        full.append(torch.cat([prompt_ids, c], dim=0))

    input_ids, attn = pad_and_stack(full, pad_id=pad_id)
    input_ids = input_ids.to(device)
    attn = attn.to(device)

    logits = model(input_ids=input_ids, attention_mask=attn).logits
    logp_all = torch.log_softmax(logits, dim=-1)

    lp_list = []
    for k, c in enumerate(cont_ids_list):
        Lc = int(c.numel())
        if Lc == 0:
            lp_list.append(torch.zeros((), device=device, dtype=torch.float32))
            continue
        pred = logp_all[k, Lp - 1: Lp + Lc - 1, :]  # (Lc, V)
        tgt = c.to(device).unsqueeze(1)             # (Lc, 1)
        lp = pred.gather(1, tgt).squeeze(1).sum()
        lp_list.append(lp)

    return torch.stack(lp_list, dim=0)  # (K,)

# -----------------------------
# 4) one estimator update (one forward + one backward)
# -----------------------------
def update_three_estimators(
    model, tok,
    prompt_ids: torch.Tensor,
    cont_ids_list: List[torch.Tensor],
    rewards: np.ndarray,    # (K,)
    V_oracle: float,        # scalar
    stat_naive: RunningTraceCov,
    stat_grpo: RunningTraceCov,
    stat_orac: RunningTraceCov,
):
    device = model.device
    G = torch.tensor(rewards, dtype=torch.float32, device=device)  # (K,)
    lp = batched_logprob_continuations(model, tok, prompt_ids, cont_ids_list)  # (K,)
    Gbar = G.mean()
    Vx = torch.tensor(float(V_oracle), dtype=torch.float32, device=device)

    K = int(G.numel())
    grpo_factor = (K / (K - 1)) if K > 1 else 1.0

    loss_naive = - (G * lp).mean()
    loss_grpo  = - (grpo_factor * (G - Gbar) * lp).mean()
    loss_orac  = - ((G - Vx) * lp).mean()

    model.zero_grad(set_to_none=True)
    loss_naive.backward(retain_graph=True)
    stat_naive.update_from_grads()

    model.zero_grad(set_to_none=True)
    loss_grpo.backward(retain_graph=True)
    stat_grpo.update_from_grads()

    model.zero_grad(set_to_none=True)
    loss_orac.backward()
    stat_orac.update_from_grads()

    model.zero_grad(set_to_none=True)

# -----------------------------
# 5) main experiment
# -----------------------------
def run_experiment_cond_only(
    model_name: str,
    oracle_table_path: str,
    device: str,
    dtype: str,
    seed: int,
    T: int,
    M: int,
    K_list: List[int],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    gen_bs: int,
    subsample_prompts: Optional[int] = None,
    save_json: Optional[str] = None,  
):
    np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    rows = load_oracle_table(oracle_table_path, limit=subsample_prompts, seed=seed)
    np_rng.shuffle(rows)
    rows = rows[:T]



    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)

    # no parameter updates, but we do backward() for gradient stats
    model.eval()

    cond_curves = {"naive": [], "grpo": [], "oracleV": []}

    for K in tqdm(K_list, desc="K sweep", dynamic_ncols=True):
        tqdm.write(f"\n=== Running K={K}, T={T}, M={M}, N=M*K={M*K} ===")

        # accumulate average across prompts
        cond_sum = {"naive": 0.0, "grpo": 0.0, "oracleV": 0.0}
        prompt_results = []

        pbar_t = tqdm(enumerate(rows), total=len(rows), desc=f"prompts@K={K}", dynamic_ncols=True, leave=False)
        for ti, row in pbar_t:
            messages = row["prompt"]
            gold = int(row["ground_truth"])
            V_oracle = float(row["V_oracle"])

            # per-prompt stats: across m = 1..M
            stat_t = {
                "naive": RunningTraceCov(model.parameters(), device=device),
                "grpo": RunningTraceCov(model.parameters(), device=device),
                "oracleV": RunningTraceCov(model.parameters(), device=device),
            }
    


            # generate N=M*K once then split
            N = int(M) * int(K)
            prompt_ids, cont_all, texts_all = sample_N(
                model, tok, messages,
                N=N,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                gen_bs=gen_bs,
            )

            for m in range(M):
                s = m * K
                e = (m + 1) * K
                cont_ids_list = cont_all[s:e]
                texts = texts_all[s:e]
                rewards = np.array([reward_01_from_text(tx, gold) for tx in texts], dtype=np.float32)

                update_three_estimators(model, tok, prompt_ids, cont_ids_list, rewards, V_oracle, stat_t["naive"], stat_t["grpo"], stat_t["oracleV"]
)

            # accumulate trace(cov) for this prompt
            tr_naive = stat_t["naive"].trace_cov()
            tr_grpo = stat_t["grpo"].trace_cov()
            tr_oracleV = stat_t["oracleV"].trace_cov()
            
            cond_sum["naive"]   += tr_naive
            cond_sum["grpo"]    += tr_grpo
            cond_sum["oracleV"] += tr_oracleV
            
            # 保存每个prompt的详细结果
            prompt_results.append({
                "prompt_index": ti,
                "prompt": messages,
                "gold": gold,
                "V_oracle": V_oracle,
                "K": K,
                "tr_cov": {
                    "naive": tr_naive,
                    "grpo": tr_grpo,
                    "oracleV": tr_oracleV
                }
            })

            if (ti + 1) % max(1, T // 10) == 0 or (ti + 1) == T:
                pbar_t.set_postfix({
                    "cond_naive": f"{(cond_sum['naive']/(ti+1)):.2e}",
                    "cond_grpo":  f"{(cond_sum['grpo']/(ti+1)):.2e}",
                    "cond_orac":  f"{(cond_sum['oracleV']/(ti+1)):.2e}",
                })

        # average across prompts
        cond = {k: cond_sum[k] / T for k in cond_sum}
        cond_curves["naive"].append(cond["naive"])
        cond_curves["grpo"].append(cond["grpo"])
        cond_curves["oracleV"].append(cond["oracleV"])
        

        if save_json:

            base_name = save_json.replace('.json', '')
            incremental_file = f"{base_name}_K{K}_detailed.json"
            with open(incremental_file, 'w') as f:
                json.dump({
                    "K": K,
                    "prompt_results": prompt_results,
                    "avg_traces": cond
                }, f, indent=2)
            tqdm.write(f"Saved K={K} detailed results to: {incremental_file}")
        
        del prompt_results
        
        tqdm.write("Final conditional traces (avg over prompts):")
        tqdm.write(f" cond-on-x: {cond}")

    return cond_curves

# -----------------------------
# 6) CLI + save plot (no show)
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument(
        "--oracle_table",
        type=str,
        required=True,
        help="path to oracle JSONL or Parquet",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--T", type=int, default=500, help="number of prompts to use (<= oracle table size)")
    ap.add_argument("--M", type=int, default=16, help="repeats per prompt for conditional variance (>=2)")
    ap.add_argument("--K_list", type=int, nargs="+", default=[4,8,16,32,64], help="list of K (group size) values to sweep")

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=32)

    ap.add_argument("--gen_bs", type=int, default=64,
                    help="max num_return_sequences per generate() for one prompt (chunks of min(M*K, gen_bs))")

    ap.add_argument("--subsample_prompts", type=int, default=None,
                    help="optional: subsample oracle table before taking first T (for quick debug)")
    ap.add_argument("--save_json", type=str, required=True,
                    help="output directory for JSON results")
    ap.add_argument("--plot_path", type=str, default=None,
                    help="output directory for plot (required unless --skip_plot is enabled)")
    ap.add_argument("--skip_plot", action="store_true",
                    help="skip saving the plot and only save JSON outputs")

    args = ap.parse_args()

    if not args.skip_plot and not args.plot_path:
        ap.error("--plot_path is required unless --skip_plot is enabled")

    # Build output filenames from the model argument.
    import os
    from datetime import datetime
    
    # Sanitize model name for filename usage.
    model_name_clean = args.model.replace("/", "_").replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build concrete output file paths.
    json_filename = f"tracecov_{model_name_clean}_{timestamp}.json"
    plot_filename = f"tracecov_{model_name_clean}_{timestamp}.png"
    
    args.save_json = os.path.join(args.save_json, json_filename)
    if not args.skip_plot:
        args.plot_path = os.path.join(args.plot_path, plot_filename)

    cond_curves = run_experiment_cond_only(
        model_name=args.model,
        oracle_table_path=args.oracle_table,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        T=args.T,
        M=args.M,
        K_list=args.K_list,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        gen_bs=args.gen_bs,
        subsample_prompts=args.subsample_prompts,
        save_json=args.save_json,  # Pass path for incremental per-K saving.
    )

    # Save main aggregate results (average curves for plotting).
    payload = {
        "K_list": args.K_list,
        "conditional_on_x": cond_curves,  # Average traces for plotting.
        "config": vars(args),
        "note": "Detailed results for each K are saved separately as *_K{K}_detailed.json files"
    }
    with open(args.save_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results json to: {args.save_json}")

    if args.skip_plot:
        print("Skipping plot generation (--skip_plot enabled).")
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(args.K_list, cond_curves["naive"], marker="o", label="Naive (G)")
        plt.plot(args.K_list, cond_curves["grpo"], marker="s", label="GRPO (K/(K-1))*(G-Gbar)")
        plt.plot(args.K_list, cond_curves["oracleV"], marker="^", label="Oracle V(x) from table")
        plt.xlabel("K (group size)")
        plt.ylabel("E_t[ trace(cov) across m | X_t ]")
        plt.title("Full-parameter trace(cov): conditional on X")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot_path, dpi=200)
        plt.close()
        print(f"Saved conditional plot to: {args.plot_path}")

if __name__ == "__main__":
    main()

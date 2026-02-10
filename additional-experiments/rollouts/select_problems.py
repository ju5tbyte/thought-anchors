"""
Select problems where the model achieves 25%~75% accuracy over N resampled attempts.

Usage:
    python select_problems.py -m deepseek/deepseek-r1-distill-qwen-14b -ds strategyqa
    python select_problems.py -m deepseek/deepseek-r1-distill-qwen-14b -ds math -nr 50

Output:
    JSON file with per-problem accuracy and a comma-separated include_problems string
    ready to pass to generate_rollouts.py via --include_problems.
"""

import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

# Add both this script's directory (for utils_datasets) and the project root (for utils, prompts)
_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parent.parent
sys.path.insert(0, str(_this_dir))
sys.path.insert(0, str(_root_dir))

from utils_datasets import get_dataset_config

parser = argparse.ArgumentParser(description="Select problems by model accuracy range")
parser.add_argument("-m", "--model", type=str, default="deepseek/deepseek-r1-distill-qwen-14b")
parser.add_argument("-ds", "--dataset", type=str, default="strategyqa", choices=["math", "strategyqa"])
parser.add_argument("-sp", "--split", type=str, default="train", choices=["train", "test"])
parser.add_argument("-np", "--num_problems", type=int, default=None, help="Number of problems to evaluate (None = all)")
parser.add_argument("-nr", "--num_rollouts", type=int, default=50, help="Number of samples per problem")
parser.add_argument("-t", "--temperature", type=float, default=0.6)
parser.add_argument("-tp", "--top_p", type=float, default=0.95)
parser.add_argument("-mt", "--max_tokens", type=int, default=4096)
parser.add_argument("-ng", "--num_gpus", type=int, default=1)
parser.add_argument("-lo", "--low", type=float, default=0.25, help="Lower accuracy bound (inclusive)")
parser.add_argument("-hi", "--high", type=float, default=0.75, help="Upper accuracy bound (inclusive)")
parser.add_argument("-o", "--output", type=str, default=None, help="Output JSON file path (default: auto-named)")
parser.add_argument("-s", "--seed", type=int, default=42)
parser.add_argument("-ty", "--type", type=str, default=None, help="Problem type filter (math only)")
parser.add_argument("-l", "--level", type=str, default=None, help="Problem level filter (math only)")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
from vllm import LLM, SamplingParams

model_name = args.model.replace("deepseek/", "deepseek-ai/")
print(f"Loading model: {model_name}")
llm = LLM(
    model=model_name,
    dtype="float16",
    tensor_parallel_size=args.num_gpus,
    max_model_len=args.max_tokens,
)
sampling_params = SamplingParams(
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.max_tokens,
)
print("Model loaded.")

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
dataset_config = get_dataset_config(args.dataset)
load_fn = dataset_config["load_problems"]

if args.dataset == "math":
    problems: List[Tuple[int, Dict]] = load_fn(
        problem_type=args.type,
        level=args.level,
        num_problems=args.num_problems,
        split=args.split,
    )
else:
    problems: List[Tuple[int, Dict]] = load_fn(
        num_problems=args.num_problems,
        split=args.split,
    )

print(f"Loaded {len(problems)} problems.")

# ---------------------------------------------------------------------------
# For each problem, generate num_rollouts completions and compute accuracy
# ---------------------------------------------------------------------------
results = []

for problem_idx, problem in problems:
    prompt = dataset_config["build_base_prompt"](problem)
    prompts = [prompt] * args.num_rollouts

    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    for req_output in outputs:
        text = req_output.outputs[0].text
        answer = dataset_config["extract_answer"](text)
        if answer and problem.get("gt_answer"):
            if dataset_config["check_answer"](answer, problem["gt_answer"]):
                correct += 1

    accuracy = correct / args.num_rollouts
    results.append({
        "problem_id": problem_idx,
        "accuracy": accuracy,
        "correct": correct,
        "total": args.num_rollouts,
        "in_range": args.low <= accuracy <= args.high,
    })
    print(f"Problem {problem_idx}: accuracy={accuracy:.2f} ({correct}/{args.num_rollouts})"
          + (" [SELECTED]" if args.low <= accuracy <= args.high else ""))

# ---------------------------------------------------------------------------
# Filter and summarise
# ---------------------------------------------------------------------------
selected = [r for r in results if r["in_range"]]
selected_ids = [r["problem_id"] for r in selected]
include_problems_str = ",".join(str(i) for i in selected_ids)

print(f"\n{'='*60}")
print(f"Total problems evaluated : {len(results)}")
print(f"Selected ({args.low:.0%}~{args.high:.0%}) : {len(selected)}")
print(f"\n--include_problems string:\n{include_problems_str}")

# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------
if args.output is None:
    model_short = args.model.split("/")[-1]
    output_path = _this_dir / f"selected_problems_{args.dataset}_{model_short}_nr{args.num_rollouts}.json"
else:
    output_path = Path(args.output)

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump({
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "num_rollouts": args.num_rollouts,
        "low": args.low,
        "high": args.high,
        "num_selected": len(selected),
        "include_problems_str": include_problems_str,
        "problems": results,
    }, f, indent=2)

print(f"\nSaved to: {output_path}")

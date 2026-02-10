"""
Dataset abstraction layer for thought-anchors rollout experiments.

To add a new dataset (e.g., GSM8K):
  1. Implement load_<dataset>_problems(), _<dataset>_build_base_prompt(), etc.
  2. Define <DATASET>_CONFIG
  3. Add it to DATASET_REGISTRY
  4. Add the dataset name to generate_rollouts.py and analyze_rollouts.py argparse choices
"""

import re
import sys
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Add root project directory to path to import from root utils.py and prompts.py
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Shared type alias
# ---------------------------------------------------------------------------

# ProblemDict schema (all datasets must produce this shape):
#   problem:     str   - question / problem text (required)
#   gt_answer:   str   - ground-truth answer string (required)
#   gt_solution: str   - ground-truth solution / explanation (optional, "" if absent)
#   level:       str   - difficulty level (optional, "" if absent)
#   type:        str   - problem category (optional, dataset name if absent)
#   dataset:     str   - dataset identifier, e.g. "math" / "strategyqa" (required)

ProblemDict = Dict  # TypedDict not used to keep Python 3.7 compatibility


# ---------------------------------------------------------------------------
# DAG prompts
# ---------------------------------------------------------------------------

DAG_PROMPT_SQA = """
You are an expert in interpreting how language models answer yes/no commonsense reasoning questions using multi-step reasoning. Your task is to analyze a Chain-of-Thought (CoT) reasoning trace, broken into discrete text chunks, and label each chunk with:

1. **function_tags**: One or more labels that describe what this chunk is *doing* functionally in the reasoning process.

2. **depends_on**: A list of earlier chunk indices that this chunk directly depends on — meaning it uses information, results, or logic introduced in those earlier chunks.

This annotation will be used to build a dependency graph and perform causal analysis, so please be precise and conservative: only mark a chunk as dependent on another if its reasoning clearly uses a previous step's result or idea.

---

### Function Tags (you may assign multiple per chunk if appropriate):

1. `problem_setup`:
    Parsing or rephrasing the yes/no question (initial reading or comprehension).

2. `fact_retrieval`:
    Recalling specific factual knowledge needed to answer (without immediate deduction).

3. `logical_deduction`:
    Deriving a conclusion from retrieved facts or prior reasoning steps toward the yes/no answer.

4. `uncertainty_management`:
    Expressing confusion, re-evaluating, reconsidering, or backtracking on a prior step.

5. `result_consolidation`:
    Aggregating or synthesizing multiple reasoning steps before emitting the final answer.

6. `final_answer_emission`:
    The explicit yes/no answer — either the <answer> tag itself or an immediately preceding statement that directly names the answer.

7. `self_checking`:
    Verifying or re-confirming a previous reasoning step or factual claim.

8. `unknown`:
    Use only if the chunk does not fit any of the above tags or is purely stylistic.

---

### depends_on Instructions:

For each chunk, include a list of earlier chunk indices that the reasoning in this chunk *uses*. For example:
- If Chunk 9 deduces a conclusion based on facts recalled in Chunks 4 and 5, then `depends_on: [4, 5]`
- If Chunk 12 verifies a claim from Chunk 10, then `depends_on: [10]`
- If there's no clear dependency (e.g. a general recall), use an empty list: `[]`

Important Notes:
- Make sure to include all dependencies for each chunk.
- Include both long-range and short-range dependencies.
- Do NOT forget about long-range dependencies.
- Try to be as comprehensive as possible.
- Make sure there is always a path from earlier chunks (e.g. problem_setup and/or fact_retrieval) to the final answer.

---

### Output Format:

Return a single dictionary with one entry per chunk, where each entry has:
- the chunk index (as the key, converted to a string),
- a dictionary with:
    - `"function_tags"`: list of tag strings
    - `"depends_on"`: list of chunk indices, converted to strings

Here's the expected format:

```language=json
{{
    "4": {{
    "function_tags": ["fact_retrieval"],
    "depends_on": []
    }},
    "5": {{
    "function_tags": ["logical_deduction"],
    "depends_on": ["4"]
    }},
    "9": {{
    "function_tags": ["result_consolidation"],
    "depends_on": ["4", "5"]
    }},
    "10": {{
    "function_tags": ["final_answer_emission"],
    "depends_on": ["9"]
    }}
}}
```

Here is the yes/no question:

[PROBLEM]
{problem_text}

Here is the full Chain of Thought, broken into chunks:

[CHUNKS]
{full_chunked_text}

Now label each chunk with function tags and dependencies.
"""

# Registry of DAG prompts per dataset. Loaded by analyze_rollouts.py.
DATASET_DAG_PROMPTS: Dict[str, str] = {
    "strategyqa": DAG_PROMPT_SQA,
    # math: use the default DAG_PROMPT from prompts.py
}


# ---------------------------------------------------------------------------
# MATH dataset
# ---------------------------------------------------------------------------


def _math_build_base_prompt(problem: ProblemDict) -> str:
    return (
        f"Solve this math problem step by step. "
        f"You MUST put your final answer in \\boxed{{}}. "
        f"Problem: {problem['problem']} Solution: \n<think>\n"
    )


def _math_build_rollout_prompt(
    problem: ProblemDict, prefix_without_chunk: str
) -> str:
    return (
        f"Solve this math problem step by step. "
        f"You MUST put your final answer in \\boxed{{}}. "
        f"Problem: {problem['problem']} Solution: \n<think>\n{prefix_without_chunk}"
    )


def _math_extract_answer(solution_text: str) -> str:
    from utils import extract_boxed_answers

    answers = extract_boxed_answers(solution_text)
    return answers[0] if answers else ""


def _math_check_answer(answer: str, gt_answer: str) -> bool:
    from utils import check_answer

    return check_answer(answer, gt_answer)


def load_math_problems(
    problem_type: Optional[str] = None,
    level: Optional[str] = None,
    num_problems: Optional[int] = None,
    split: str = "train",
    include_problems: Optional[List[int]] = None,
) -> List[Tuple[int, ProblemDict]]:
    """Thin wrapper around root utils.load_math_problems."""
    from utils import load_math_problems as _load

    return _load(
        problem_type=problem_type,
        level=level,
        num_problems=num_problems,
        split=split,
        include_problems=include_problems,
    )


MATH_CONFIG: Dict = {
    "name": "math",
    "build_base_prompt": _math_build_base_prompt,
    "build_rollout_prompt": _math_build_rollout_prompt,
    "extract_answer": _math_extract_answer,
    "check_answer": _math_check_answer,
    "load_problems": load_math_problems,
}


# ---------------------------------------------------------------------------
# StrategyQA dataset
# ---------------------------------------------------------------------------


def _sqa_build_base_prompt(problem: ProblemDict) -> str:
    return (
        f"Answer the following yes/no question with step-by-step reasoning. "
        f"You MUST end your response with <answer>yes</answer> or <answer>no</answer>. "
        f"Question: {problem['problem']}\n<think>\n"
    )


def _sqa_build_rollout_prompt(
    problem: ProblemDict, prefix_without_chunk: str
) -> str:
    return (
        f"Answer the following yes/no question with step-by-step reasoning. "
        f"You MUST end your response with <answer>yes</answer> or <answer>no</answer>. "
        f"Question: {problem['problem']}\n<think>\n{prefix_without_chunk}"
    )


def _sqa_extract_answer(solution_text: str) -> str:
    """Extract yes/no from <answer>yes</answer> tag; fallback to keyword scan."""
    match = re.search(r"<answer>\s*(yes|no)\s*</answer>", solution_text.lower())
    if match:
        return match.group(1)
    # Fallback: scan last 3 lines for yes/no keyword
    last_lines = solution_text.strip().lower().split("\n")[-3:]
    for line in reversed(last_lines):
        if re.search(r"\byes\b", line):
            return "yes"
        if re.search(r"\bno\b", line):
            return "no"
    return ""


def _sqa_check_answer(answer: str, gt_answer: str) -> bool:
    return answer.strip().lower() == gt_answer.strip().lower()


def load_strategyqa_problems(
    num_problems: Optional[int] = None,
    split: str = "train",
    include_problems: Optional[List[int]] = None,
) -> List[Tuple[int, ProblemDict]]:
    """
    Load problems from the StrategyQA dataset (wics/strategy-qa).

    Returns List[Tuple[int, ProblemDict]] where ProblemDict has:
        problem:      str         - yes/no question text
        gt_answer:    str         - "yes" or "no"
        gt_solution:  str         - facts joined by newlines (empty string if absent)
        level:        str         - "" (StrategyQA has no difficulty level)
        type:         str         - "strategyqa"
        dataset:      str         - "strategyqa"
        facts:        List[str]   - raw supporting facts (preserved for analysis)
        decomposition: List[str]  - raw sub-questions (preserved for analysis)
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "json",
            data_files="https://raw.githubusercontent.com/wicsaax/strategy-qa/main/strategyQA_train.json",
        )
    except Exception as e:
        print(f"Error loading StrategyQA dataset: {e}")
        return []

    available_splits = list(dataset.keys())
    if split not in available_splits:
        print(
            f"Split '{split}' not found. Available: {available_splits}. Using first split."
        )
        split = available_splits[0]

    split_data = dataset[split]

    indexed_problems: List[Tuple[int, ProblemDict]] = []
    for i, item in enumerate(split_data):
        # Convert bool answer to "yes"/"no"
        raw_answer = item.get("answer", item.get("Answer", False))
        if isinstance(raw_answer, bool):
            gt_answer = "yes" if raw_answer else "no"
        else:
            gt_answer = str(raw_answer).strip().lower()

        facts: List[str] = item.get("facts", []) or []
        decomposition: List[str] = item.get("decomposition", []) or []
        gt_solution = "\n".join(facts) if facts else ""

        problem_dict: ProblemDict = {
            "problem": item["question"],
            "gt_answer": gt_answer,
            "gt_solution": gt_solution,
            "level": "",
            "type": "strategyqa",
            "dataset": "strategyqa",
            "facts": facts,
            "decomposition": decomposition,
        }
        indexed_problems.append((i, problem_dict))

    # Filter by explicit problem IDs
    if include_problems is not None:
        include_set = set(include_problems)
        indexed_problems = [
            (i, p) for i, p in indexed_problems if i in include_set
        ]
    elif num_problems is not None and num_problems < len(indexed_problems):
        indexed_problems = random.sample(indexed_problems, num_problems)

    print(
        f"Loaded {len(indexed_problems)} StrategyQA problems from split '{split}'."
    )
    return indexed_problems


STRATEGYQA_CONFIG: Dict = {
    "name": "strategyqa",
    "build_base_prompt": _sqa_build_base_prompt,
    "build_rollout_prompt": _sqa_build_rollout_prompt,
    "extract_answer": _sqa_extract_answer,
    "check_answer": _sqa_check_answer,
    "load_problems": load_strategyqa_problems,
}


# ---------------------------------------------------------------------------
# Registry & accessor
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict] = {
    "math": MATH_CONFIG,
    "strategyqa": STRATEGYQA_CONFIG,
    # Future datasets:
    # "gsm8k": GSM8K_CONFIG,
    # "arc": ARC_CONFIG,
}


def get_dataset_config(dataset_name: str) -> Dict:
    """Return the DatasetConfig for the given dataset name."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]

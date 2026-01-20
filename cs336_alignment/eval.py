
import json
import os
import pandas as pd
import pyarrow.parquet as pq
from typing import Callable, List
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams
from collections import defaultdict


def evaluate_vllm(
vllm_model: LLM,
reward_fn: Callable[[str, str], dict[str, float]],
prompts: List[str],
ground_truth: List[str],
eval_sampling_params: SamplingParams
) -> None:

    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    metrics = defaultdict(int)
    examples = defaultdict(list)
    # Print the outputs.
    for output, gt in zip(outputs, ground_truth):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        reward = reward_fn(generated_text, gt)
        metrics["total_count"] += 1
        metrics["total_reward"] += reward["reward"]
        if reward["format_reward"] == 1 and reward["answer_reward"]  == 1:
            metrics["f1_a1"] += 1
            examples["f1_a1"].append((gt, generated_text))
        elif reward["format_reward"] == 1 and reward["answer_reward"]  == 0:
            metrics["f1_a0"] += 1
            examples["f1_a0"].append((gt, generated_text))
        elif reward["format_reward"] == 0 and reward["answer_reward"]  == 0:
            metrics["f0_a0"] += 1
            examples["f0_a0"].append((gt, generated_text))

    metrics["avg_reward"] = metrics["total_reward"]/metrics["total_count"]
    print(f"Avg Reward: {metrics["avg_reward"]}")
    print(metrics)

    # Save metrics and examples to disk
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    metrics_path = os.path.join(results_dir, "metrics.json")
    examples_path = os.path.join(results_dir, "examples.json")

    # Convert tuples in examples to serializable form
    serializable_examples = {k: [list(pair) for pair in v] for k, v in examples.items()}

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(dict(metrics), f, indent=2, ensure_ascii=False)

    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(serializable_examples, f, indent=2, ensure_ascii=False)






if __name__ == "__main__":
    # Choose dataset: "gsm8k" or "math"
    dataset = "math"  # Change to "gsm8k" to use GSM8K dataset
    
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"])

    baseline_prompt = ""
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "r1_zero.prompt")
    with open(prompt_path, 'r', encoding='utf-8') as file:
        baseline_prompt = file.read()

    prompts = []
    ground_truth = []
    
    if dataset == "gsm8k":
        # load and format questions with baseline prompt from GSM8K dataset
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "gsm8k", "test.jsonl")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_object = json.loads(line)
                formatted_prompt = baseline_prompt.replace("{question}", json_object["question"])
                prompts.append(formatted_prompt)
                ground_truth.append(json_object["answer"])
    
    elif dataset == "math":
        # load and format questions with baseline prompt from MATH dataset
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "MATH", "train-00000-of-00001-7320a6f3aba8ebd2.parquet")
        
        # Try to load from symlink path, fallback to cache if symlink is broken
        if not os.path.exists(data_path) or not os.path.isfile(data_path):
            # Fallback to the actual cache location
            data_path = os.path.expanduser("~/.cache/huggingface/hub/datasets--qwedsacf--competition_math/blobs/2325458edc03d786939ee9e1e5795efb9e2480247b6e1ed2c51f41bea7369c6a")
        
        # Load parquet file
        df = pq.read_table(data_path).to_pandas()
        
        # Extract problems and solutions
        for _, row in df.iterrows():
            problem = row["problem"]
            solution = row["solution"]
            formatted_prompt = baseline_prompt.replace("{question}", problem)
            prompts.append(formatted_prompt)
            ground_truth.append(solution)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'gsm8k' or 'math'.")

    evaluate_vllm(llm, r1_zero_reward_fn, prompts, ground_truth, sampling_params)    
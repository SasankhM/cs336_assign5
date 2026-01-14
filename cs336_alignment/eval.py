
import json
import os
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
    total = defaultdict(int)
    total_reward = 0
    total_count = 0
    examples = defaultdict(list)
    # Print the outputs.
    for output, gt in zip(outputs, ground_truth):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(generated_text, gt)
        reward = reward_fn(generated_text, gt)
        total_count += 1
        total_reward += reward["reward"]
        if reward["format_reward"] == 1 and reward["answer_reward"]  == 1:
            total["f1_a1"] += 1
            examples["f1_a1"].append(generated_text)
        elif reward["format_reward"] == 1 and reward["answer_reward"]  == 0:
            total["f1_a0"] += 1
            examples["f1_a0"].append(generated_text)
        elif reward["format_reward"] == 0 and reward["answer_reward"]  == 0:
            total["f0_a0"] += 1
            examples["f0_a0"].append(generated_text)

    print(f"Avg Reward: {total_reward/total_count}")
    print(total)
    print(examples["f1_a1"][-1])
    print(examples["f1_a0"][0])
    print(examples["f0_a0"][0])






if __name__ == "__main__":
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"])

    baseline_prompt = ""
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "r1_zero.prompt")
    with open(prompt_path, 'r', encoding='utf-8') as file:
        baseline_prompt = file.read()

    prompts = []
    ground_truth = []
    # load and format questions with baseline prompt
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "gsm8k", "test.jsonl")
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
                json_object = json.loads(line)
                formatted_prompt = baseline_prompt.replace("{question}", json_object["question"])
                prompts.append(formatted_prompt)
                ground_truth.append(json_object["answer"])

    evaluate_vllm(llm, r1_zero_reward_fn, prompts, ground_truth, sampling_params)    
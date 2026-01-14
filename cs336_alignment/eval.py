
import json
from typing import Callable, List
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams


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
    # Print the outputs.
    for output, ground_truth in zip(outputs, ground_truth):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        reward = reward_fn(generated_text, ground_truth)
        print(reward)



if __name__ == "__main__":
    llm = LLM(model="models/qwen.safetensors")
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"])

    baseline_prompt = ""
    with open("cs336_alignment/prompts/r1_zero.prompt", 'r', encoding='utf-8') as file:
        baseline_prompt = file.read()

    prompts = []
    ground_truth = []
    # load and format questions with baseline prompt
    with open("data/gsm8k/test.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
                json_object = json.loads(line)
                formatted_prompt = baseline_prompt.replace("{question}", json_object["question"])
                prompts.append(formatted_prompt)
                ground_truth.append(json_object["answer"])

    evaluate_vllm(llm, r1_zero_reward_fn, prompts, ground_truth, sampling_params)    
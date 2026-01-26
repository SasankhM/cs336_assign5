

from cs336_alignment.sft_helper import (
    MATHSFTDataset, 
    get_response_log_probs, 
    load_eval_data_for_validation, 
    sft_microbatch_train_step, 
    tokenize_prompt_and_output,
    log_generations
)
import torch 
import torch.optim as optim
import argparse
import os
from unittest.mock import patch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
from tqdm import tqdm 
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import wandb





def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    13
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def train_model(num_examples: int | None, 
                lr: float, 
                batch_size: int,
                gradient_accumulation_steps: int,
                num_epochs: int,
                output_dir: str,
                device: str):
    
    # Initialize wandb with informative run name
    num_examples_str = str(num_examples) if num_examples is not None else "full"
    run_name = f"sft_n{num_examples_str}_lr{lr}_bs{batch_size}x{gradient_accumulation_steps}_ep{num_epochs}"
    
    wandb.init(
        project="sft-math",
        name=run_name,
        config={
            "num_examples": num_examples,
            "lr": lr,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "num_epochs": num_epochs,
        }
    )
    
    # Setup wandb metrics
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    tokenizer = AutoTokenizer.from_pretrained("/data/a5-alignment/models/Qwen2.5-Math-1.5B")

    # Load training dataset from Arrow file
    train_data_path = "data/MATH/train/data-00000-of-00001.arrow"
    prompt_template_path = "cs336_alignment/prompts/r1_zero.prompt"
    train_dataset = MATHSFTDataset(train_data_path, tokenizer, num_examples, prompt_template_path)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    # Initialize vLLM on separate GPU for evaluation
    llm = init_vllm("/data/a5-alignment/models/Qwen2.5-Math-1.5B", "cuda:1", seed=42)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    # Load evaluation data
    eval_data_path = "data/MATH/eval/data-00000-of-00001.arrow"
    prompt_template_path = "cs336_alignment/prompts/r1_zero.prompt"
    eval_prompts, eval_ground_truths = load_eval_data_for_validation(eval_data_path, prompt_template_path)
    
    # Create output directory for generations log
    os.makedirs(output_dir, exist_ok=True)
    generations_log_path = os.path.join(output_dir, "generations.jsonl")

    # Training loop with global step tracking
    global_step = 0
    eval_step = 0
    optimizer.zero_grad()  # Clear any stale gradients
    
    for epoch in range(num_epochs):
        model.train()  # Ensure model is in training mode at start of each epoch
        for idx, train_batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            input_ids = train_batch["input_ids"].to(device)
            labels = train_batch["labels"].to(device)
            response_mask = train_batch["response_mask"].to(device)
            
            # Forward Pass
            log_probs_dict = get_response_log_probs(model, input_ids, labels)

            # Backward Pass
            loss, _ = sft_microbatch_train_step(
                log_probs_dict["log_probs"], 
                response_mask,
                gradient_accumulation_steps
            )

            # Optimizer step after accumulation
            if (idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log training metrics
                wandb.log({
                    "train/loss": loss.item(),
                    "train_step": global_step,
                })

        # Evaluate at end of each epoch
        print(f"Epoch {epoch+1} complete. Running validation...")
        accuracy, eval_metrics = validate_one_epoch(
            model, 
            llm, 
            eval_prompts, 
            eval_ground_truths,
            sampling_params, 
            r1_zero_reward_fn,
            generations_log_path,
            global_step
        )
        eval_step += 1
        
        # Log eval metrics
        wandb.log({
            "eval/accuracy": accuracy,
            "eval/avg_reward": eval_metrics["avg_reward"],
            "eval_step": eval_step,
        })
        
        print(f"Epoch {epoch+1} - Validation accuracy: {accuracy:.4f}")

    # Save final model
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)
    
    wandb.finish()




def validate_one_epoch(model, llm, prompts, ground_truths, sampling_params, reward_fn, 
                       generations_log_path, global_step):
    """
    Validate the model on evaluation prompts using vLLM for generation.
    
    Returns:
        accuracy: float, the fraction of correct answers
        metrics: dict with additional metrics like avg_reward, response lengths
    """
    model.eval()
    load_policy_into_vllm_instance(model, llm)

    # Generate responses
    outputs = llm.generate(prompts, sampling_params)
    
    # Score and collect metrics
    total_correct = 0
    total_reward = 0.0
    generations = []
    metadata_list = []
    
    response_lengths = []
    response_lengths_correct = []
    response_lengths_incorrect = []
    
    for output, gt in zip(outputs, ground_truths):
        response = output.outputs[0].text
        generations.append(response)
        reward_result = reward_fn(response, gt)
        
        total_reward += reward_result["reward"]
        response_len = len(response)
        response_lengths.append(response_len)
        
        if reward_result["reward"] == 1.0:
            total_correct += 1
            response_lengths_correct.append(response_len)
        else:
            response_lengths_incorrect.append(response_len)
        
        # Collect metadata for logging
        metadata_list.append({
            "ground_truth": gt,
            "format_reward": reward_result["format_reward"],
            "answer_reward": reward_result["answer_reward"],
            "reward": reward_result["reward"],
            "response_length": response_len,
            "global_step": global_step,
        })
    
    accuracy = total_correct / len(prompts)
    avg_reward = total_reward / len(prompts)
    avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    avg_response_length_correct = sum(response_lengths_correct) / len(response_lengths_correct) if response_lengths_correct else 0
    avg_response_length_incorrect = sum(response_lengths_incorrect) / len(response_lengths_incorrect) if response_lengths_incorrect else 0
    
    # Restore training mode
    model.train()
    
    # Log generations using the helper function
    log_generations(
        prompts=prompts,
        generations=generations,
        output_path=generations_log_path,
        metadata=metadata_list,
        append=True  # Append to keep history across evaluations
    )
    
    metrics = {
        "avg_reward": avg_reward,
        "avg_response_length": avg_response_length,
        "avg_response_length_correct": avg_response_length_correct,
        "avg_response_length_incorrect": avg_response_length_incorrect,
    }
    
    model.train()  # Switch back to training mode
    return accuracy, metrics









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT training on MATH dataset")
    parser.add_argument("--num_examples", type=int, default=None, 
                        help="Number of training examples (None for full dataset)")
    parser.add_argument("--lr", type=float, default=1e-6, 
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                        help="Number of gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=5, 
                        help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="results/sft/", 
                        help="Output directory for model and logs")
    args = parser.parse_args()
    
    train_model(
        num_examples=args.num_examples,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        device="cuda:0"
    )


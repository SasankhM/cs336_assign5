from typing import Any, Callable, Literal
import json
import os

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import torch.nn.functional as F




def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase) -> dict[str, Tensor]: 
    """ Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for
        other tokens (prompt or padding) """ 

    prompt_tokens = tokenizer(prompt_strs, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length)
    output_tokens = tokenizer(output_strs, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length)

    batch_size = len(prompt_strs)
    
    # Compute actual lengths (not padded) for all examples
    prompt_lens = [(prompt_tokens['input_ids'][i] != tokenizer.pad_token_id).sum().item() for i in range(batch_size)]
    output_lens = [(output_tokens['input_ids'][i] != tokenizer.pad_token_id).sum().item() for i in range(batch_size)]
    max_prompt_and_output_lens = max(prompt_len + output_len for prompt_len, output_len in zip(prompt_lens, output_lens))

    # build initial answer
    input_ids = torch.full((batch_size, max_prompt_and_output_lens-1), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_prompt_and_output_lens-1), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros(batch_size, max_prompt_and_output_lens-1, dtype=torch.long)

    for i in range(batch_size):
        prompt_len, output_len = prompt_lens[i], output_lens[i]
        prompt_and_output = torch.cat((prompt_tokens["input_ids"][i, :prompt_len], output_tokens["input_ids"][i, :output_len]), dim=0)
        
        # Pad prompt_and_output to max_prompt_and_output_lens
        if prompt_and_output.size(0) < max_prompt_and_output_lens:
            padding = torch.full((max_prompt_and_output_lens - prompt_and_output.size(0),), tokenizer.pad_token_id, dtype=torch.long)
            prompt_and_output = torch.cat((prompt_and_output, padding), dim=0)

        input_ids[i, :] = prompt_and_output[:-1]
        labels[i, :] = prompt_and_output[1:]
        mask[i, prompt_len-1:prompt_len+output_len-1] = 1

    
    output = {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": mask
    }

    return output 
        



def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
    containing unnormalized logits.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
    prediction.
    """

    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdims=True) # log sum trick to get log probs
    probs = torch.exp(log_probs) # use the log version to just get probs

    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy




def get_response_log_probs(
model: PreTrainedModel,
input_ids: torch.Tensor,
labels: torch.Tensor,
return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:

    """
    Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
    and in inference mode if gradients should not be computed).

        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
    response tokens as produced by your tokenization method.

        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
    tokenization method.
    
        return_token_entropy: bool If True, also return per-token entropy by calling
    compute_entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
            log pθ(xt | x<t).
            "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
            for each position (present only if return_token_entropy=True).
    """ 
    output = {}

    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    output["log_probs"] = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    if return_token_entropy:
        output["token_entropy"] = compute_entropy(logits)

    return output



def masked_normalize(
tensor: torch.Tensor,
mask: torch.Tensor,
normalize_constant: float,
dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask
    == 1.
    Args:
        tensor: torch.Tensor The tensor to sum and normalize.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float the constant to divide by for normalization.
        dim: int | None the dimension to sum along before normalization. If None, sum over all
        dimensions.
    Returns:
        torch.Tensor the normalized sum, where masked elements (mask == 0) don’t contribute to
    the sum.
    """ 

    masked_tensor = torch.where(mask > 0, tensor, 0)
    output = torch.sum(masked_tensor, dim=dim) / normalize_constant
    return output 



def sft_microbatch_train_step(
policy_log_probs: torch.Tensor,
response_mask: torch.Tensor,
gradient_accumulation_steps: int,
normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
    SFT policy being trained.

        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
        prompt/padding.

        gradient_accumulation_steps Number of microbatches per optimizer step.

        normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.
    
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]. loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
    this so we can log it.

        metadata Dict with metadata from the underlying loss call, and any other statistics you
    might want to log
    """ 
    loss = masked_normalize(tensor=-policy_log_probs, 
                            mask=response_mask, 
                            normalize_constant=normalize_constant * policy_log_probs.shape[0] * gradient_accumulation_steps) # normalize per example (batch_size) and also average out by gradient accumulation steps

    loss.backward()

    return loss, {}


def log_generations(
    prompts: list[str],
    generations: list[str],
    output_path: str,
    metadata: list[dict[str, Any]] | None = None,
    append: bool = False,
) -> None:
    """
    Log model generations to a file in JSONL format.
    
    Args:
        prompts: List of prompt strings that were used for generation.
        generations: List of generated text strings corresponding to each prompt.
        output_path: Path to the output file where generations will be saved (JSONL format).
        metadata: Optional list of dictionaries containing additional metadata for each
            generation (e.g., rewards, metrics, model info). Must have same length as prompts.
        append: If True, append to existing file. If False, overwrite existing file.
    
    Returns:
        None. Writes generations to the specified file.
    
    Example:
        >>> prompts = ["What is 2+2?"]
        >>> generations = ["4"]  # 2. The response generated by the SFT/RL model
        >>> metadata = [{
        ...     # 1. The input prompt (already logged via prompts parameter)
        ...     # 2. The response generated by the SFT/RL model (already logged via generations parameter)
        ...     "ground_truth": "4",  # 3. The ground-truth answer
        ...     "format_reward": 1.0,  # 4. The reward information
        ...     "answer_reward": 1.0,
        ...     "reward": 1.0,
        ...     "avg_token_entropy": 2.5,  # 5. The average token entropy of the response
        ...     "avg_response_length": 1.0,  # 6. The average response length
        ...     "avg_response_length_correct": 1.0,  # Average response length for correct responses
        ...     "avg_response_length_incorrect": 0.0  # Average response length for incorrect responses
        ... }]
        >>> log_generations(prompts, generations, "output.jsonl", metadata=metadata)
        >>> # The logged JSONL entry will contain:
        >>> # {
        >>> #   "prompt": "What is 2+2?",  # 1. The input prompt
        >>> #   "generation": "4",  # 2. The response generated by the SFT/RL model
        >>> #   "ground_truth": "4",  # 3. The ground-truth answer
        >>> #   "format_reward": 1.0,  # 4. The reward information
        >>> #   "answer_reward": 1.0,
        >>> #   "reward": 1.0,
        >>> #   "avg_token_entropy": 2.5,  # 5. The average token entropy of the response
        >>> #   "avg_response_length": 1.0,  # 6. The average response length
        >>> #   "avg_response_length_correct": 1.0,  # Average response length for correct responses
        >>> #   "avg_response_length_incorrect": 0.0  # Average response length for incorrect responses
        >>> # }
    """
    if len(prompts) != len(generations):
        raise ValueError(f"prompts and generations must have the same length. Got {len(prompts)} prompts and {len(generations)} generations.")
    
    if metadata is not None and len(metadata) != len(prompts):
        raise ValueError(f"metadata must have the same length as prompts. Got {len(metadata)} metadata entries and {len(prompts)} prompts.")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Write mode: 'a' for append, 'w' for write (overwrite)
    mode = 'a' if append else 'w'
    
    with open(output_path, mode, encoding='utf-8') as f:
        for i, (prompt, generation) in enumerate(zip(prompts, generations)):
            entry = {
                "prompt": prompt,
                "generation": generation,
            }
            
            # Add metadata if provided
            if metadata is not None:
                entry.update(metadata[i])
            
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    

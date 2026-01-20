#!/usr/bin/env python3
"""
Download Qwen2.5-Math-1.5B model to the specified path.
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

# Model ID on HuggingFace
model_id = "Qwen/Qwen2.5-Math-1.5B"

# Target directory
target_dir = Path("/data/a5-alignment/models/Qwen2.5-Math-1.5B")

# Create parent directory if it doesn't exist
target_dir.parent.mkdir(parents=True, exist_ok=True)

print(f"Downloading model {model_id} to {target_dir}...")
print("This may take a while depending on your internet connection...")
print("The model is approximately 3GB in size.\n")

# Download entire model repository
snapshot_download(
    repo_id=model_id,
    local_dir=str(target_dir),
    local_dir_use_symlinks=False,  # Use actual files, not symlinks
)

print(f"\n✓ Model successfully downloaded to {target_dir}")

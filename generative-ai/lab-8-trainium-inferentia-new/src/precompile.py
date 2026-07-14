# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Pre-compile Neuron graphs for the training script.

This script runs on ml.trn1.2xlarge using neuron_parallel_compile to trace and
compile all graph segments needed by train.py. It uses the REAL training data
(passed via SageMaker input channels) to ensure all graph shapes match what
the actual training job will encounter.

The compiled .neff files are saved to the Neuron compile cache, which is then
uploaded to S3. Subsequent training jobs on trn1.2xlarge can load these cached
graphs and skip compilation entirely.

Usage (via SageMaker Training job):
    python precompile.py --model_id Qwen/Qwen3-0.6B --max_seq_length 512 \
        --batch_size 2 --num_workers 1 --cache_dir /opt/ml/model
"""

import os
import sys
import argparse
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compile Neuron graphs for training")

    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel compilation workers (nproc_per_node for training)")
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
                        help="Directory to save compiled cache for upload to S3")
    # SageMaker input channels (real training data)
    parser.add_argument("--training_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", ""),
                        help="Path to training data (from S3 input channel)")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info(f"Pre-compilation arguments: {args}")

    # Install optimum-neuron[training]==0.4.2 for NeuronSFTTrainer support
    # The DLC ships with 0.4.1 which lacks NeuronSFTTrainer/NeuronSFTConfig
    logger.info("Installing optimum-neuron[training]==0.4.2...")
    subprocess.run(
        ["pip", "install", "optimum-neuron[training]==0.4.2", "--quiet"],
        check=True,
    )
    logger.info("Installation complete.")

    # Set the Neuron compile cache directory
    os.environ["NEURON_COMPILE_CACHE_URL"] = args.cache_dir
    os.makedirs(args.cache_dir, exist_ok=True)

    # Determine training data path
    if args.training_dir and os.path.isdir(args.training_dir):
        train_data_dir = args.training_dir
        logger.info(f"Using real training data from: {train_data_dir}")
    else:
        # Fallback: create dummy data if no input channel provided
        logger.warning("No training data input channel found, creating dummy dataset...")
        train_data_dir = "/tmp/dummy_train_data"
        os.makedirs(train_data_dir, exist_ok=True)
        _create_dummy_dataset(args.max_seq_length, train_data_dir)

    # Build the train.py command that neuron_parallel_compile will trace
    train_script = os.path.join(os.path.dirname(__file__), "train.py")

    train_cmd = [
        "torchrun",
        f"--nproc_per_node={args.num_workers}",
        train_script,
        "--model_id", args.model_id,
        "--max_seq_length", str(args.max_seq_length),
        "--batch_size", str(args.batch_size),
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--epochs", "1",
        "--training_dir", train_data_dir,
        "--model_dir", "/tmp/model_output",
        "--output_data_dir", "/tmp/output_data",
    ]

    # Run neuron_parallel_compile which traces the graph and compiles all modules
    compile_cmd = ["neuron_parallel_compile"] + train_cmd

    logger.info(f"Running neuron_parallel_compile...")
    logger.info(f"Command: {' '.join(compile_cmd)}")

    result = subprocess.run(
        compile_cmd,
        env={**os.environ, "NEURON_COMPILE_CACHE_URL": args.cache_dir},
        capture_output=True,
        text=True,
    )

    # Print stdout/stderr regardless of result for debugging
    if result.stdout:
        logger.info(f"neuron_parallel_compile stdout:\n{result.stdout[-5000:]}")
    if result.stderr:
        logger.error(f"neuron_parallel_compile stderr:\n{result.stderr[-5000:]}")

    if result.returncode != 0:
        logger.error(f"neuron_parallel_compile failed with exit code {result.returncode}")
        sys.exit(1)

    # Verify cache was populated
    cache_files = []
    for root, dirs, files in os.walk(args.cache_dir):
        for f in files:
            if f.endswith(".neff"):
                cache_files.append(os.path.join(root, f))

    logger.info(f"Compilation complete! Generated {len(cache_files)} .neff files")
    for f in cache_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        logger.info(f"  {os.path.basename(f)}: {size_mb:.1f} MB")

    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info("Cache will be uploaded to S3 automatically by SageMaker.")


def _create_dummy_dataset(seq_length, output_dir, num_samples=8):
    """Fallback: create minimal dummy dataset matching raw Dolly format."""
    from datasets import Dataset

    # Create dummy data matching Dolly schema (instruction, context, response)
    dataset = Dataset.from_dict({
        "instruction": ["What is AWS?" for _ in range(num_samples)],
        "context": ["" for _ in range(num_samples)],
        "response": ["AWS stands for Amazon Web Services." for _ in range(num_samples)],
    })
    dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    main()

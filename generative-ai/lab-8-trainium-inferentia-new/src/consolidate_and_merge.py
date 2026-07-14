# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Consolidate Neuron distributed checkpoint and merge LoRA adapter with base model.

NeuronSFTTrainer saves adapters in Neuron distributed format (adapter_shards/).
This script:
1. Installs optimum-neuron[training]==0.4.2
2. Consolidates the sharded checkpoint into standard PEFT format
3. Merges the adapter with the base model
4. Saves the full merged model to /opt/ml/model (uploaded to S3 by SageMaker)

Usage (via SageMaker Training job on trn1.2xlarge or any Neuron instance):
    python consolidate_and_merge.py --model_id Qwen/Qwen3-0.6B --adapter_dir /opt/ml/input/data/adapter
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
    parser = argparse.ArgumentParser(description="Consolidate and merge LoRA adapter")

    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace base model ID")
    parser.add_argument("--adapter_dir", type=str,
                        default=os.environ.get("SM_CHANNEL_ADAPTER", "/opt/ml/input/data/adapter"),
                        help="Path to the Neuron sharded adapter checkpoint")
    parser.add_argument("--model_dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
                        help="Output directory for merged model")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info(f"Arguments: {args}")

    # Install optimum-neuron for consolidation support
    logger.info("Installing optimum-neuron[training]==0.4.2...")
    subprocess.run(
        ["pip", "install", "optimum-neuron[training]==0.4.2", "--quiet"],
        check=True,
    )
    logger.info("Installation complete.")

    # Step 1: Consolidate Neuron distributed checkpoint
    consolidated_dir = "/tmp/adapter_consolidated"
    os.makedirs(consolidated_dir, exist_ok=True)

    logger.info(f"Consolidating checkpoint from: {args.adapter_dir}")
    logger.info(f"Adapter contents: {os.listdir(args.adapter_dir)}")

    # SageMaker input channels download files but don't extract them
    # If we find a model.tar.gz, extract it first
    tar_path = os.path.join(args.adapter_dir, "model.tar.gz")
    if os.path.isfile(tar_path):
        import tarfile
        extract_dir = "/tmp/adapter_extracted"
        os.makedirs(extract_dir, exist_ok=True)
        logger.info(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        args.adapter_dir = extract_dir
        logger.info(f"Extracted contents: {os.listdir(extract_dir)}")

    # List adapter_default contents for debugging
    adapter_default_dir = os.path.join(args.adapter_dir, "adapter_default")
    if os.path.isdir(adapter_default_dir):
        logger.info(f"adapter_default contents: {os.listdir(adapter_default_dir)}")
    else:
        # Maybe the tar extracted without the top-level directory
        logger.info("adapter_default not found at expected path, scanning...")
        for root, dirs, files in os.walk(args.adapter_dir):
            for f in files[:20]:
                logger.info(f"  {os.path.join(root, f)}")
            if len(files) > 20:
                logger.info(f"  ... and {len(files) - 20} more files")
                break

    # The adapter_dir contains the extracted tar.gz which has adapter_default/ inside
    # optimum-cli neuron consolidate expects the directory containing adapter_default/
    try:
        subprocess.run(
            ["optimum-cli", "neuron", "consolidate", args.adapter_dir, consolidated_dir],
            check=True,
            capture_output=True,
            text=True,
        )
        # Copy adapter_config.json from source if not present in consolidated dir
        src_config = os.path.join(args.adapter_dir, "adapter_default", "adapter_config.json")
        dst_config = os.path.join(consolidated_dir, "adapter_config.json")
        if os.path.isfile(src_config) and not os.path.isfile(dst_config):
            import shutil
            shutil.copy2(src_config, dst_config)
            logger.info(f"Copied adapter_config.json from source to {consolidated_dir}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"optimum-cli consolidate failed: {e.stderr}")
        logger.info("Attempting direct extraction for single-rank checkpoint...")

        # For nproc_per_node=1, the checkpoint has only 1 shard — extract directly
        # Load the single shard and save as standard PEFT format
        import torch
        from peft import LoraConfig

        # Find the shards directory dynamically
        shards_dir = None
        for root, dirs, files in os.walk(args.adapter_dir):
            if any(f.endswith(".pt") and "info" not in f and "tensors" not in f for f in files):
                shards_dir = root
                break

        if shards_dir is None:
            raise FileNotFoundError(f"Could not find .pt shard files anywhere under {args.adapter_dir}")

        shard_files = [f for f in os.listdir(shards_dir) if f.endswith(".pt") and "info" not in f and "tensors" not in f]
        logger.info(f"Found shard files in {shards_dir}: {shard_files}")

        # Load the state dict from the single shard
        shard_path = os.path.join(shards_dir, shard_files[0])
        state_dict = torch.load(shard_path, map_location="cpu")
        logger.info(f"Loaded state dict with {len(state_dict)} keys")

        # Save in standard PEFT format
        os.makedirs(consolidated_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(consolidated_dir, "adapter_model.bin"))

        # Create adapter_config.json from training config
        # Read from the metadata if available
        metadata_file = os.path.join(adapter_default_dir, "adapter_shards", "mp_metadata_pp_rank_0.json")
        import json as json_mod
        if os.path.isfile(metadata_file):
            with open(metadata_file) as f:
                metadata = json_mod.load(f)
            logger.info(f"Metadata: {metadata}")

        # Create a minimal adapter_config.json
        adapter_config_dict = {
            "base_model_name_or_path": args.model_id,
            "bias": "none",
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "r": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA",
        }
        with open(os.path.join(consolidated_dir, "adapter_config.json"), "w") as f:
            json_mod.dump(adapter_config_dict, f, indent=2)

        logger.info(f"Created consolidated adapter: {os.listdir(consolidated_dir)}")

    logger.info(f"Consolidated adapter: {os.listdir(consolidated_dir)}")

    # Step 2: Load base model and merge
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig

    logger.info(f"Loading base model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    logger.info("Loading consolidated LoRA adapter...")
    adapter_config = PeftConfig.from_pretrained(consolidated_dir)
    model = PeftModel.from_pretrained(model, consolidated_dir, config=adapter_config)

    logger.info("Merging adapter into base model...")
    model = model.merge_and_unload()

    # Step 3: Save merged model
    logger.info(f"Saving merged model to: {args.model_dir}")
    model.save_pretrained(args.model_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.model_dir)

    # Ensure torch_dtype is set in config.json (required by inference DLC)
    import json as json_mod
    config_path = os.path.join(args.model_dir, "config.json")
    with open(config_path) as f:
        config = json_mod.load(f)
    if config.get("torch_dtype") is None:
        config["torch_dtype"] = "float16"
        with open(config_path, "w") as f:
            json_mod.dump(config, f, indent=2)
        logger.info("Set torch_dtype=float16 in config.json")

    logger.info(f"Merged model files: {os.listdir(args.model_dir)}")
    logger.info("Done! Merged model will be uploaded to S3 by SageMaker.")


if __name__ == "__main__":
    main()

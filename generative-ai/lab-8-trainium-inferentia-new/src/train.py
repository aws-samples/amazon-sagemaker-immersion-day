# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Fine-tune a causal language model on AWS Trainium using NeuronSFTTrainer + LoRA.

Uses NeuronModelForCausalLM (Neuron-native training model loader) for proper
bf16 handling and flash attention support on Trainium hardware.

Based on the official optimum-neuron Qwen3 fine-tuning tutorial:
https://huggingface.co/docs/optimum-neuron/training_tutorials/finetune_qwen3

Usage:
    torchrun --nproc_per_node=1 train.py --model_id Qwen/Qwen3-0.6B ...
"""

import os
import sys
import argparse
import logging

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from peft import LoraConfig

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with LoRA on Trainium")

    # Model
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")

    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=1)

    # SageMaker environment paths
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--training_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", ""))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    args, _ = parser.parse_known_args()
    return args


def format_dolly(example):
    """Format a single Dolly dataset sample into instruction-response text.

    NeuronSFTTrainer calls this per-example (not batched).
    """
    instruction = f"### Instruction\n{example['instruction']}"
    context = (
        f"### Context\n{example['context']}"
        if len(example["context"]) > 0
        else None
    )
    response = f"### Answer\n{example['response']}"
    return "\n\n".join([part for part in [instruction, context, response] if part is not None])


def main():
    args = parse_args()
    logger.info(f"Training arguments: {args}")

    # Import optimum-neuron modules (installed by run_training.py wrapper)
    from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
    from optimum.neuron.models.training import NeuronModelForCausalLM

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure training arguments (needed for NeuronModelForCausalLM)
    training_args = NeuronTrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        bf16=True,
        logging_dir=os.path.join(args.output_data_dir, "logs"),
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to="none",
        overwrite_output_dir=True,
        lr_scheduler_type="cosine",
    )

    # Load model using Neuron-native training model loader
    # This handles bf16 casting correctly for Trainium
    logger.info(f"Loading model: {args.model_id}")
    model = NeuronModelForCausalLM.from_pretrained(
        args.model_id,
        training_args.trn_config,
        dtype=torch.bfloat16,
    )

    # LoRA configuration — following official optimum-neuron Qwen3 tutorial
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load dataset
    logger.info(f"Loading training data from: {args.training_dir}")
    train_dataset = load_from_disk(args.training_dir)

    # NeuronSFTConfig — combines NeuronTrainingArguments + TRL SFTConfig
    sft_config = NeuronSFTConfig(
        max_length=args.max_seq_length,
        packing=False,
        **training_args.to_dict(),
    )

    # NeuronSFTTrainer — handles LoRA application, tokenization, and Neuron optimizations
    trainer = NeuronSFTTrainer(
        args=sft_config,
        model=model,
        peft_config=lora_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        formatting_func=format_dolly,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")

    # Save the LoRA adapter
    trainer.save_model()
    tokenizer.save_pretrained(args.model_dir)
    logger.info(f"Model saved to: {args.model_dir}")


if __name__ == "__main__":
    main()

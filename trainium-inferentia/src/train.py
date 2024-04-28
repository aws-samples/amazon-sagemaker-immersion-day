# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import sys
import torch
import random
import argparse
import evaluate
import importlib
import traceback
import subprocess
import transformers

from huggingface_hub import login
from datasets import load_from_disk
from transformers import AutoTokenizer
from optimum.neuron import NeuronTrainer as Trainer
from optimum.neuron import NeuronTrainingArguments as TrainingArguments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_sen_len", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--zero_1", type=bool, default=False)
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--collator", type=str, default="DefaultDataCollator")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", type=bool, default=True)

    # hugging face hub
    parser.add_argument("--hf_token", type=str, default=None)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_neurons", type=str, default=os.environ["SM_NUM_NEURONS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--eval_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", None))

    parser.add_argument('--checkpoints-path', type=str, help="Path where we'll save the cache", default='/opt/ml/checkpoints')

    args, _ = parser.parse_known_args()
    os.makedirs(args.checkpoints_path, exist_ok=True)

    if not args.hf_token is None and len(args.hf_token) > 0:
        print("HF token defined. Logging in...")
        login(token=args.hf_token)
        
        cmd = f"optimum-cli neuron cache set {os.environ['CUSTOM_CACHE_REPO']}"
        subprocess.check_call(cmd.split(' '))

    Collator = eval(f"transformers.{args.collator}")
    AutoModel = eval(f"transformers.AutoModel{'For' + args.task if len(args.task) > 0 else ''}")

    train_dataset=load_from_disk(args.training_dir)
    eval_dataset=load_from_disk(args.eval_dir) if not args.eval_dir is None else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_sen_len

    data_collator = Collator(return_tensors="pt")
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True) # TODO: add a hyperparameter with model params

    training_args = TrainingArguments(
        evaluation_strategy="epoch" if not args.eval_dir is None else "no",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        num_train_epochs=args.epochs,
        output_dir=args.checkpoints_path,
        overwrite_output_dir=True,
        tensor_parallel_size=args.tensor_parallel_size,
        zero_1=args.zero_1,

        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size if not args.eval_dir is None else None,
        logging_dir=f"{args.output_data_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        save_steps=1000,
        save_strategy="steps",
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    # save artifacts that will be uploaded to S3
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Workshop Preparation: Upload dataset + launch Neuron precompile job.

This script:
1. Loads and subsets the Dolly dataset
2. Uploads it to S3
3. Launches a SageMaker training job for neuron_parallel_compile
4. Prints the job name and exits (non-blocking)

Usage:
    python scripts/workshop_prep.py

Check job status:
    aws sagemaker describe-training-job --training-job-name <job-name> --region us-east-2
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from datasets import load_dataset
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, SourceCode, InputData
from sagemaker.train.distributed import Torchrun
from sagemaker.core.shapes import OutputDataConfig

from config import MODEL_ID, AWS_REGION


# --- Configuration ---
DATASET_ID = "databricks/databricks-dolly-15k"
NUM_TRAIN_SAMPLES = 1000
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 2
LORA_R = 16
LORA_ALPHA = 16
GRADIENT_ACCUMULATION_STEPS = 4
NUM_WORKERS = 1  # nproc_per_node=1 (single process gets full 32GB host RAM for compiler)

S3_PREFIX = "lab8-trainium-inferentia"
TRAINING_IMAGE_TAG = "huggingface-pytorch-training-neuronx:2.8.0-transformers4.55.4-neuronx-py310-sdk2.26.0-ubuntu22.04"
PROCESSING_INSTANCE_TYPE = "ml.trn1.2xlarge"


def get_session_and_role():
    """Initialize SageMaker session and resolve execution role."""
    boto_session = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = Session(boto_session)
    try:
        role = get_execution_role()
    except ValueError:
        role = None

    if role is None or "AmazonSageMaker-ExecutionRole" not in role:
        account_id = boto_session.client("sts").get_caller_identity()["Account"]
        role = f"arn:aws:iam::{account_id}:role/service-role/AmazonSageMaker-ExecutionRole-20260525T102256"

    return sagemaker_session, role, boto_session


def upload_dataset(sagemaker_session, bucket):
    """Load, subset, and upload the raw Dolly dataset to S3."""
    print(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train")
    dataset = dataset.shuffle(seed=42).select(range(NUM_TRAIN_SAMPLES))

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")

    local_path = "datasets/train"
    os.makedirs(local_path, exist_ok=True)
    dataset.save_to_disk(local_path)

    train_s3_uri = sagemaker_session.upload_data(
        local_path, bucket=bucket, key_prefix=f"{S3_PREFIX}/datasets/train"
    )
    print(f"Dataset uploaded to: {train_s3_uri}")
    return train_s3_uri


def launch_precompile_job(sagemaker_session, role, bucket, region, train_s3_uri):
    """Launch the neuron_parallel_compile job (non-blocking)."""
    model_short = MODEL_ID.split("/")[-1]
    cache_s3_uri = f"s3://{bucket}/{S3_PREFIX}/neuron-cache/{model_short}"
    training_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/{TRAINING_IMAGE_TAG}"

    print(f"\nLaunching precompile job...")
    print(f"  Model: {MODEL_ID}")
    print(f"  Instance: {PROCESSING_INSTANCE_TYPE}")
    print(f"  Cache destination: {cache_s3_uri}")

    compiler = ModelTrainer(
        training_image=training_image,
        source_code=SourceCode(
            source_dir="src",
            entry_script="precompile.py",
        ),
        compute=Compute(
            instance_type=PROCESSING_INSTANCE_TYPE,
            instance_count=1,
            volume_size_in_gb=100,
        ),
        role=role,
        base_job_name="neuron-precompile",
        output_data_config=OutputDataConfig(s3_output_path=cache_s3_uri),
        distributed=Torchrun(process_count_per_node=NUM_WORKERS),
        hyperparameters={
            "model_id": MODEL_ID,
            "max_seq_length": MAX_SEQ_LENGTH,
            "batch_size": BATCH_SIZE,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "num_workers": NUM_WORKERS,
            "cache_dir": "/opt/ml/model",
        },
        environment={
            "MALLOC_ARENA_MAX": "64",
            "NEURON_FUSE_SOFTMAX": "1",
            "NEURON_CC_FLAGS": "--model-type=transformer --distribution-strategy=llm-training",
        },
    )

    # Launch non-blocking — train() with wait=False is not available in ModelTrainer,
    # so we use the underlying _start_training method pattern
    compiler.train(
        input_data_config=[
            InputData(channel_name="train", data_source=train_s3_uri),
        ],
        wait=False,
    )

    job_name = compiler._latest_training_job.training_job_name
    return job_name, cache_s3_uri


def main():
    import argparse as _ap
    parser = _ap.ArgumentParser()
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip dataset upload (use if data already in S3)")
    cli_args = parser.parse_args()

    print("=" * 60)
    print("Workshop Preparation: Dataset + Neuron Precompilation")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Region: {AWS_REGION}")
    print()

    sagemaker_session, role, boto_session = get_session_and_role()
    bucket = sagemaker_session.default_bucket()
    region = AWS_REGION

    print(f"Bucket: {bucket}")
    print(f"Role: {role}")
    print()

    # Step 1: Upload dataset
    if cli_args.skip_upload:
        print("-" * 40)
        print("Step 1: Skipping upload (--skip-upload)")
        print("-" * 40)
        train_s3_uri = f"s3://{bucket}/{S3_PREFIX}/datasets/train"
        print(f"Using existing data at: {train_s3_uri}")
    else:
        print("-" * 40)
        print("Step 1: Upload dataset to S3")
        print("-" * 40)
        train_s3_uri = upload_dataset(sagemaker_session, bucket)
    print()

    # Step 2: Launch precompile job (non-blocking)
    print("-" * 40)
    print("Step 2: Launch precompile job")
    print("-" * 40)
    job_name, cache_s3_uri = launch_precompile_job(
        sagemaker_session, role, bucket, region, train_s3_uri
    )

    print()
    print("=" * 60)
    print("Precompile job launched (non-blocking)!")
    print("=" * 60)
    print(f"  Job name: {job_name}")
    print(f"  Cache S3:  {cache_s3_uri}")
    print()
    print("Monitor with:")
    print(f"  aws sagemaker describe-training-job --training-job-name {job_name} --region {region}")
    print()
    print("Once completed, run: python scripts/fine_tuning.py")


if __name__ == "__main__":
    main()

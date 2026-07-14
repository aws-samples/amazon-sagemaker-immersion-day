#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Launch LoRA fine-tuning job on AWS Trainium.

This script:
1. Configures a ModelTrainer with the Neuron DLC and pre-compiled cache
2. Launches the SageMaker training job
3. Prints the job name and exits (non-blocking)

Prerequisites:
- Dataset uploaded to S3 (run scripts/workshop_prep.py first)
- Neuron compile cache in S3 (precompile job must have completed)

Usage:
    python scripts/fine_tuning.py

Check job status:
    aws sagemaker describe-training-job --training-job-name <job-name> --region us-east-2

View logs:
    aws logs get-log-events --log-group-name /aws/sagemaker/TrainingJobs \
        --log-stream-name <job-name>/algo-1-* --region us-east-2
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, SourceCode, InputData
from sagemaker.train.distributed import Torchrun
from sagemaker.core.shapes import OutputDataConfig

from config import MODEL_ID, AWS_REGION


# --- Configuration ---
MAX_SEQ_LENGTH = 512
S3_PREFIX = "lab8-trainium-inferentia"
TRAINING_IMAGE_TAG = "huggingface-pytorch-training-neuronx:2.8.0-transformers4.55.4-neuronx-py310-sdk2.26.0-ubuntu22.04"


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

    return sagemaker_session, role


def main():
    print("=" * 60)
    print("Fine-Tune LLM on AWS Trainium with LoRA")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Region: {AWS_REGION}")
    print()

    sagemaker_session, role = get_session_and_role()
    bucket = sagemaker_session.default_bucket()
    region = AWS_REGION
    model_short = MODEL_ID.split("/")[-1]

    print(f"Bucket: {bucket}")
    print(f"Role: {role}")
    print()

    # S3 paths
    train_s3_uri = f"s3://{bucket}/{S3_PREFIX}/datasets/train"
    cache_s3_uri = f"s3://{bucket}/{S3_PREFIX}/neuron-cache/{model_short}"
    output_s3_uri = f"s3://{bucket}/{S3_PREFIX}/output"
    training_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/{TRAINING_IMAGE_TAG}"

    print(f"Training data: {train_s3_uri}")
    print(f"Neuron cache:  {cache_s3_uri}")
    print(f"Output:        {output_s3_uri}")
    print()

    # Launch training job
    trainer = ModelTrainer(
        training_image=training_image,
        source_code=SourceCode(
            source_dir="src",
            entry_script="train.py",
        ),
        compute=Compute(
            instance_type="ml.trn1.2xlarge",
            instance_count=1,
            volume_size_in_gb=256,
        ),
        role=role,
        base_job_name="qwen25-lora-trn1",
        output_data_config=OutputDataConfig(s3_output_path=output_s3_uri),
        distributed=Torchrun(process_count_per_node=1),
        hyperparameters={
            "model_id": MODEL_ID,
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 5e-5,
            "max_seq_length": MAX_SEQ_LENGTH,
            "lora_r": 16,
            "lora_alpha": 16,
            "gradient_accumulation_steps": 4,
        },
        environment={
            "MALLOC_ARENA_MAX": "64",
            "NEURON_FUSE_SOFTMAX": "1",
            "NEURON_CC_FLAGS": "--model-type=transformer --distribution-strategy=llm-training",
            "NEURON_COMPILE_CACHE_URL": cache_s3_uri,
        },
    )

    print("Launching training job...")
    trainer.train(
        input_data_config=[
            InputData(channel_name="train", data_source=train_s3_uri),
        ],
        wait=False,
    )

    job_name = trainer._latest_training_job.training_job_name
    print()
    print("=" * 60)
    print("Training job launched (non-blocking)!")
    print("=" * 60)
    print(f"  Job name: {job_name}")
    print(f"  Model:    {MODEL_ID}")
    print(f"  Instance: ml.trn1.2xlarge")
    print()
    print("Monitor with:")
    print(f"  aws sagemaker describe-training-job --training-job-name {job_name} --region {region}")
    print()
    print("View logs:")
    print(f"  aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix {job_name} --follow --region {region}")


if __name__ == "__main__":
    main()

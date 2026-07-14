#!/usr/bin/env python3
"""
test_inference_inf2.py — Run inference on inf2 EC2 instance using compiled neffs.

Run this AFTER compile_on_inf2.sh has completed successfully.
It loads the compiled model from the local directory and runs test prompts.

Usage (on inf2.xlarge EC2 via SSM):
    source /home/ubuntu/inf2_env/bin/activate
    python3 /home/ubuntu/test_inference_inf2.py

Or point to a specific compiled model directory:
    python3 /home/ubuntu/test_inference_inf2.py --model_dir /tmp/inf2-compile/compiled
"""

import os
import sys
import time
import argparse

os.environ["NEURON_RT_NUM_CORES"] = "2"


def parse_args():
    parser = argparse.ArgumentParser(description="Test inference on inf2 with compiled model")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/tmp/inf2-compile/compiled",
        help="Path to compiled model directory (default: /tmp/inf2-compile/compiled)",
    )
    parser.add_argument(
        "--s3_uri",
        type=str,
        default=None,
        help="S3 URI to compiled model.tar.gz (downloads and extracts if --model_dir doesn't exist)",
    )
    return parser.parse_args()


def download_from_s3(s3_uri, model_dir):
    """Download and extract model from S3 if local dir doesn't exist."""
    import subprocess
    import tarfile

    os.makedirs(model_dir, exist_ok=True)
    tar_path = "/tmp/compiled-model.tar.gz"

    print(f"Downloading {s3_uri}...")
    subprocess.run(["aws", "s3", "cp", s3_uri, tar_path, "--region", "us-east-2"], check=True)

    print(f"Extracting to {model_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(model_dir)

    os.remove(tar_path)
    print(f"Model files: {os.listdir(model_dir)}")


def main():
    args = parse_args()
    model_dir = args.model_dir

    # Download from S3 if needed
    if not os.path.isdir(model_dir) or not os.listdir(model_dir):
        if args.s3_uri:
            download_from_s3(args.s3_uri, model_dir)
        else:
            # Default S3 path
            default_s3 = "s3://sagemaker-us-east-2-333982362808/lab8-trainium-inferentia/models/compiled/qwen25-lora-trn1-20260710070145/inf2-native/model.tar.gz"
            print(f"Local model dir not found. Downloading from default S3 path...")
            download_from_s3(default_s3, model_dir)

    print(f"\nModel directory: {model_dir}")
    print(f"Contents: {os.listdir(model_dir)}")

    # Load model
    print("\n" + "=" * 60)
    print("Loading model with optimum.neuron pipeline...")
    print("=" * 60)

    from optimum.neuron import pipeline

    start = time.time()
    pipe = pipeline("text-generation", model_dir)
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.1f}s")

    # Test prompts
    prompts = [
        {
            "name": "Closed QA (Dolly-style)",
            "text": """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction
Based on the context provided, what are the main benefits of containerization?

### Context
Containerization packages applications with their dependencies into isolated units called containers. This ensures consistency across development, testing, and production environments. Containers are lightweight, start quickly, and can be orchestrated at scale using tools like Kubernetes.

### Response
""",
        },
        {
            "name": "Brainstorming",
            "text": """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction
Suggest 3 creative ways to reduce energy consumption in a modern office building.

### Response
""",
        },
        {
            "name": "Open QA",
            "text": """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction
What is transfer learning in machine learning, and why is it useful?

### Response
""",
        },
    ]

    print("\n" + "=" * 60)
    print("Running inference...")
    print("=" * 60)

    for i, prompt_data in enumerate(prompts, 1):
        print(f"\n--- Prompt {i}: {prompt_data['name']} ---")
        print(f"Input: {prompt_data['text'][:80]}...")

        start = time.time()
        outputs = pipe(
            prompt_data["text"],
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        latency = (time.time() - start) * 1000

        generated = outputs[0]["generated_text"]
        # Strip the input prompt
        if generated.startswith(prompt_data["text"]):
            generated = generated[len(prompt_data["text"]):]

        print(f"Latency: {latency:.0f} ms")
        print(f"Response: {generated.strip()[:500]}")

    print("\n" + "=" * 60)
    print("All prompts completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

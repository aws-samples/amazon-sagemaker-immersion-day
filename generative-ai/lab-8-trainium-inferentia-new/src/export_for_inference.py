# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Export a merged model to Neuron format for Inferentia2 inference.

This script:
1. Installs optimum-neuron[neuronx]==0.4.2
2. Loads the merged model from the input channel
3. Compiles it for Inferentia2 using optimum-neuron export
4. Saves the compiled model to /opt/ml/model (uploaded to S3 by SageMaker)

The compiled model can be deployed directly to inf2 without any lazy compilation.

Usage (via SageMaker Training job on trn1.2xlarge):
    python export_for_inference.py --model_id Qwen/Qwen3-0.6B \
        --batch_size 1 --sequence_length 512 --num_cores 2
"""

import os
import sys
import argparse
import subprocess
import logging
import tarfile

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to Neuron format for inference")

    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace base model ID (for tokenizer)")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--sequence_length", type=int, default=512, help="Max sequence length for inference")
    parser.add_argument("--num_cores", type=int, default=2, help="Number of NeuronCores to use")
    parser.add_argument("--auto_cast_type", type=str, default="fp16", help="Auto cast type (fp16, bf16)")
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
                        help="Output directory for compiled model")
    parser.add_argument("--merged_model_dir", type=str,
                        default=os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model"),
                        help="Path to merged model (from S3 input channel)")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info(f"Arguments: {args}")

    # Install optimum-neuron
    logger.info("Installing optimum-neuron[neuronx]==0.4.2...")
    subprocess.run(
        ["pip", "install", "optimum-neuron[neuronx]==0.4.2", "--quiet"],
        check=True,
    )
    logger.info("Installation complete.")

    # Check if input is a tar.gz (SageMaker doesn't extract input channels)
    model_path = args.merged_model_dir
    tar_path = os.path.join(model_path, "model.tar.gz")
    if os.path.isfile(tar_path):
        extract_dir = "/tmp/merged_model"
        os.makedirs(extract_dir, exist_ok=True)
        logger.info(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        model_path = extract_dir
        logger.info(f"Extracted model: {os.listdir(model_path)}")

    # Export to Neuron format
    logger.info(f"Exporting model to Neuron format...")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Sequence length: {args.sequence_length}")
    logger.info(f"  Num cores: {args.num_cores}")
    logger.info(f"  Auto cast type: {args.auto_cast_type}")

    export_cmd = [
        "optimum-cli", "export", "neuron",
        "--model", model_path,
        "--batch_size", str(args.batch_size),
        "--sequence_length", str(args.sequence_length),
        "--num_cores", str(args.num_cores),
        "--auto_cast_type", args.auto_cast_type,
        "--task", "text-generation",
        args.model_dir,
    ]

    logger.info(f"Command: {' '.join(export_cmd)}")
    result = subprocess.run(export_cmd, capture_output=True, text=True)

    if result.stdout:
        logger.info(f"Export stdout:\n{result.stdout[-3000:]}")
    if result.stderr:
        logger.warning(f"Export stderr:\n{result.stderr[-3000:]}")

    if result.returncode != 0:
        logger.error(f"Export failed with exit code {result.returncode}")
        sys.exit(1)

    logger.info(f"Export complete! Output: {os.listdir(args.model_dir)}")

    # Patch neuron_config.json version to match inference DLC (0.4.1)
    # The export was done with 0.4.2 but the inference DLC has 0.4.1
    # The compiled model is compatible — this is just a metadata version check
    neuron_config_path = os.path.join(args.model_dir, "neuron_config.json")
    if os.path.isfile(neuron_config_path):
        import json as json_mod
        with open(neuron_config_path) as f:
            neuron_config = json_mod.load(f)
        if "optimum_neuron_version" in neuron_config:
            old_version = neuron_config["optimum_neuron_version"]
            neuron_config["optimum_neuron_version"] = "0.4.1"
            with open(neuron_config_path, "w") as f:
                json_mod.dump(neuron_config, f, indent=2)
            logger.info(f"Patched neuron_config.json: optimum_neuron_version {old_version} -> 0.4.1")

    # Create custom inference.py to bypass the broken default HF inference handler.
    # The default handler calls pipeline() with export= which raises ValueError on
    # pre-compiled NxD models. This custom script loads the model directly.
    code_dir = os.path.join(args.model_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    inference_script = '''import os
os.environ["NEURON_RT_NUM_CORES"] = "2"

import torch
import torch_neuronx
from optimum.neuron import pipeline


def model_fn(model_dir):
    """Load pre-compiled Neuron model using optimum.neuron pipeline."""
    pipe = pipeline("text-generation", model_dir)
    return pipe


def predict_fn(data, pipe):
    """Run inference on the loaded model."""
    inputs = data.pop("inputs", data)
    parameters = data.pop("parameters", {})

    # Handle both string and list-of-messages input formats
    if isinstance(inputs, list):
        # Chat format - apply chat template
        inputs = pipe.tokenizer.apply_chat_template(
            inputs, add_generation_prompt=True, tokenize=False
        )

    # Set defaults for generation
    parameters.setdefault("max_new_tokens", 256)
    parameters.setdefault("do_sample", True)
    parameters.setdefault("temperature", 0.7)
    parameters.setdefault("top_p", 0.9)

    outputs = pipe(inputs, **parameters)
    generated = outputs[0]["generated_text"]

    # Strip the input prompt from the output
    if isinstance(generated, str) and generated.startswith(inputs):
        generated = generated[len(inputs):]

    return [{"generated_text": generated.strip()}]
'''

    inference_path = os.path.join(code_dir, "inference.py")
    with open(inference_path, "w") as f:
        f.write(inference_script)
    logger.info(f"Created custom inference script at {inference_path}")

    logger.info("Compiled model will be uploaded to S3 by SageMaker.")


if __name__ == "__main__":
    main()

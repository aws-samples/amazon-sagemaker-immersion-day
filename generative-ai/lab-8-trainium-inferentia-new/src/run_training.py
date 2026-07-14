# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Wrapper script that installs optimum-neuron 0.4.2 then launches train.py via torchrun.

SageMaker launches this as a Basic Script (not via torchrun). It:
1. Pip installs optimum-neuron[training]==0.4.2 (DLC has 0.4.1)
2. Launches torchrun --nproc_per_node=1 train.py with all hyperparameters

This avoids the stale module cache issue that occurs when pip install
happens inside a process already launched by torchrun.
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # Install optimum-neuron[training]==0.4.2
    logger.info("Installing optimum-neuron[training]==0.4.2...")
    subprocess.run(
        ["pip", "install", "optimum-neuron[training]==0.4.2", "--quiet"],
        check=True,
    )
    logger.info("Installation complete.")

    # Build the torchrun command with all hyperparameters passed to this script
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")

    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        train_script,
    ] + sys.argv[1:]

    logger.info(f"Launching training: {' '.join(cmd)}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error(f"Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()

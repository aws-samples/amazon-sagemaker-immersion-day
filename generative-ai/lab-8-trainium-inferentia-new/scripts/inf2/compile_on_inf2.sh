#!/bin/bash
# compile_on_inf2.sh — Run on an inf2.xlarge EC2 instance via SSM
#
# Downloads the merged model from S3, compiles it for Inferentia2 using
# optimum-cli export neuron, patches neuron_config.json, adds custom
# inference.py, packages as model.tar.gz, and uploads to S3.
#
# Usage (via SSM):
#   aws ssm start-session --target <instance-id> --region us-east-2
#   Then paste/run this script, or:
#   aws ssm send-command --instance-ids <id> --document-name AWS-RunShellScript \
#     --parameters commands="$(cat scripts/compile_on_inf2.sh)"

set -euo pipefail

# --- Configuration ---
REGION="us-east-2"
BUCKET="sagemaker-us-east-2-333982362808"
S3_PREFIX="lab8-trainium-inferentia"
TRAINING_JOB="qwen25-lora-trn1-20260710070145"
MODEL_ID="Qwen/Qwen3-0.6B"

# S3 paths
MERGED_MODEL_S3="s3://${BUCKET}/${S3_PREFIX}/models/merged/${TRAINING_JOB}"
COMPILED_OUTPUT_S3="s3://${BUCKET}/${S3_PREFIX}/models/compiled/${TRAINING_JOB}/inf2-native"

# Local paths
WORK_DIR="/tmp/inf2-compile"
MERGED_DIR="${WORK_DIR}/merged"
COMPILED_DIR="${WORK_DIR}/compiled"

# Compile settings
BATCH_SIZE=1
SEQUENCE_LENGTH=512
NUM_CORES=2
AUTO_CAST_TYPE="fp16"

echo "============================================"
echo "Inferentia2 Model Compilation"
echo "============================================"
echo "Model: ${MODEL_ID}"
echo "Merged model: ${MERGED_MODEL_S3}"
echo "Output: ${COMPILED_OUTPUT_S3}"
echo "Batch size: ${BATCH_SIZE}, Seq length: ${SEQUENCE_LENGTH}, Cores: ${NUM_CORES}"
echo "============================================"

# --- Setup environment ---
echo ""
echo "[1/6] Setting up Python environment..."
cd /tmp

# Activate the Neuron venv
if [ -d /home/ubuntu/inf2_env ]; then
    source /home/ubuntu/inf2_env/bin/activate
    echo "Activated: /home/ubuntu/inf2_env"
elif [ -d /opt/inf2_env ]; then
    source /opt/inf2_env/bin/activate
    echo "Activated: /opt/inf2_env"
elif [ -d /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference ]; then
    source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate
    echo "Activated: aws_neuronx_venv_pytorch_2_5_nxd_inference"
else
    echo "ERROR: No Neuron venv found. Create one first with:"
    echo "  uv venv /home/ubuntu/inf2_env --python 3.10"
    echo "  source /home/ubuntu/inf2_env/bin/activate"
    echo "  uv pip install 'optimum-neuron[neuronx]' transformers torch-neuronx neuronx-cc --extra-index-url https://pip.repos.neuron.amazonaws.com"
    exit 1
fi

pip install --quiet "optimum-neuron[neuronx]" "transformers" "torch" 2>/dev/null || true
echo "optimum-neuron version: $(pip show optimum-neuron 2>/dev/null | grep Version)"
echo "neuronx-cc version: $(pip show neuronx-cc 2>/dev/null | grep Version || echo 'not found')"

# --- Download merged model ---
echo ""
echo "[2/6] Downloading merged model from S3..."
rm -rf "${WORK_DIR}"
mkdir -p "${MERGED_DIR}" "${COMPILED_DIR}"

# Find the latest merge job output
MERGE_TAR=$(aws s3 ls "${MERGED_MODEL_S3}/" --recursive --region ${REGION} | grep "model.tar.gz" | sort | tail -1 | awk '{print $NF}')
if [ -z "${MERGE_TAR}" ]; then
    echo "ERROR: No merged model found at ${MERGED_MODEL_S3}"
    exit 1
fi

echo "Found: s3://${BUCKET}/${MERGE_TAR}"
aws s3 cp "s3://${BUCKET}/${MERGE_TAR}" "${WORK_DIR}/model.tar.gz" --region ${REGION}
echo "Extracting..."
tar -xzf "${WORK_DIR}/model.tar.gz" -C "${MERGED_DIR}"
echo "Merged model files: $(ls ${MERGED_DIR})"

# --- Compile for Inferentia2 ---
echo ""
echo "[3/6] Compiling model for Inferentia2 (this may take 10-15 minutes)..."
echo "Command: optimum-cli export neuron --model ${MERGED_DIR} --batch_size ${BATCH_SIZE} --sequence_length ${SEQUENCE_LENGTH} --num_cores ${NUM_CORES} --auto_cast_type ${AUTO_CAST_TYPE} --task text-generation ${COMPILED_DIR}"

time optimum-cli export neuron \
    --model "${MERGED_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --sequence_length ${SEQUENCE_LENGTH} \
    --num_cores ${NUM_CORES} \
    --auto_cast_type ${AUTO_CAST_TYPE} \
    --task text-generation \
    "${COMPILED_DIR}"

echo "Compiled model files: $(ls ${COMPILED_DIR})"

# --- Patch neuron_config.json ---
echo ""
echo "[4/6] Checking neuron_config.json..."
NEURON_CONFIG="${COMPILED_DIR}/neuron_config.json"
if [ -f "${NEURON_CONFIG}" ]; then
    echo "Contents:"
    cat "${NEURON_CONFIG}" | python3 -m json.tool
    
    # Check the optimum_neuron_version — if compiled natively on inf2, it should
    # match the DLC. Patch only if needed.
    COMPILED_VERSION=$(python3 -c "import json; print(json.load(open('${NEURON_CONFIG}'))['optimum_neuron_version'])")
    echo "Compiled with optimum_neuron_version: ${COMPILED_VERSION}"
    
    # Patch to 0.4.1 if the inference DLC expects it
    python3 -c "
import json
with open('${NEURON_CONFIG}') as f:
    cfg = json.load(f)
if cfg.get('optimum_neuron_version') != '0.4.1':
    cfg['optimum_neuron_version'] = '0.4.1'
    with open('${NEURON_CONFIG}', 'w') as f:
        json.dump(cfg, f, indent=2)
    print('Patched optimum_neuron_version -> 0.4.1')
else:
    print('Version already 0.4.1, no patch needed')
"
fi

# --- Add custom inference.py ---
echo ""
echo "[5/6] Adding custom inference.py..."
mkdir -p "${COMPILED_DIR}/code"
cat > "${COMPILED_DIR}/code/inference.py" << 'INFERENCE_EOF'
import os
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
INFERENCE_EOF
echo "Created ${COMPILED_DIR}/code/inference.py"

# --- Package and upload ---
echo ""
echo "[6/6] Packaging model.tar.gz and uploading to S3..."
cd "${COMPILED_DIR}"
tar -czf "${WORK_DIR}/compiled-model.tar.gz" .
echo "Archive size: $(du -h ${WORK_DIR}/compiled-model.tar.gz | cut -f1)"

aws s3 cp "${WORK_DIR}/compiled-model.tar.gz" \
    "${COMPILED_OUTPUT_S3}/model.tar.gz" \
    --region ${REGION}

echo ""
echo "============================================"
echo "DONE!"
echo "============================================"
echo "Compiled model uploaded to:"
echo "  ${COMPILED_OUTPUT_S3}/model.tar.gz"
echo ""
echo "Use this S3 URI in 02_deploy_prep.ipynb to deploy the endpoint."
echo "============================================"

# Cleanup
rm -rf "${WORK_DIR}"

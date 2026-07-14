# Lab 8 Configuration
#
# Both notebooks (01_fine-tuning, 02_deployment) read from this file.
# Workshop organizers: adjust these settings based on your environment.

# --- Model Selection ---
#
# | Model ID             | Params | Est. Training Time | Output Quality |
# |----------------------|--------|--------------------|----------------|
# | Qwen/Qwen3-1.7B     | 1.7B   | ~15-20 min         | Better         |
# | Qwen/Qwen3-0.6B     | 0.6B   | ~10 min            | Lower          |
#
# Note: Smaller models run faster but produce lower quality outputs.
# For workshops with limited time, use Qwen3-0.6B.
# For better demonstration of fine-tuning impact, use Qwen3-1.7B (default).

MODEL_ID = "Qwen/Qwen3-0.6B"

# Workshop Studio provisions resources in this region.
# Trainium/Inferentia quotas and DLC images must be available here.
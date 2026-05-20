# Serverless Fine-Tuning

Supervised fine-tuning (SFT) of a JumpStart foundation model with `SFTTrainer`,
end to end on SageMaker AI without provisioning a training cluster. The base
model is configured in [`config.py`](./config.py) and defaults to
`huggingface-reasoning-qwen3-8b`.

Run the notebooks in order:

| Notebook | What it does |
|---|---|
| [`1-prepare-data.ipynb`](./1-prepare-data.ipynb) | Streams `FreedomIntelligence/medical-o1-reasoning-SFT`, reformats into prompt/completion with `<think>` tags, splits 70/10/20, registers three datasets in the SageMaker AI Registry. |
| [`2-fine-tune-llm.ipynb`](./2-fine-tune-llm.ipynb) | Submits a serverless `SFTTrainer` job with `TrainingType.LORA`. Registers the result as a model package in your Model Registry. ~20 minutes. |
| [`3-deployment.ipynb`](./3-deployment.ipynb) | Deploys the merged fine-tuned model behind a SageMaker real-time endpoint with the DJL/LMI inference container. Closes with cleanup. |

The companion workshop pages live at
[`/generative-ai/serverless/`](https://catalog.workshops.aws/) — see the workshop
content repo for the prose framing, code skeletons, and gotchas.

## Provenance

These notebooks originated from the
[serverless-model-customization-with-sagemaker-ai](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/tree/main/workshops/serverless-model-customization-with-sagemaker-ai/lab-1-supervised-fine-tuning)
workshop in `aws-samples/generative-ai-on-amazon-sagemaker`. They have been
pulled into this repo so the notebook surface matches the workshop layout.
The deployment notebook was renumbered from `4-deployment.ipynb` to
`3-deployment.ipynb` because the evaluation step (notebook 3 upstream) is
not yet covered in the workshop pages.

# FMOps on SageMaker AI

FMOps / LLMOps for a foundation-model workload on SageMaker AI, tracked in
SageMaker Managed MLflow. First the individual FMOps building blocks run by
hand; then the same steps run as an orchestrated SageMaker Pipeline.

Run the notebooks in order:

| Notebook | What it does |
|---|---|
| [`00_fmops_examples.ipynb`](./00_fmops_examples.ipynb) | Runs the FMOps components individually (no orchestration): deploy a model to a SageMaker endpoint tracked in MLflow, evaluate it with MLflow's LLM-as-a-judge, attach an Amazon Bedrock Guardrail, and add observability with MLflow model tracing. |
| [`01_fine-tuning-pipeline.ipynb`](./01_fine-tuning-pipeline.ipynb) | Stitches those steps into a SageMaker Pipeline: preprocess → fine-tune Qwen3 4B Instruct → evaluate (ROUGE + LLM-as-a-judge) → conditionally register to MLflow and the SageMaker Model Registry. |

Supporting code:

| Path | Purpose |
|---|---|
| `steps/` | Pipeline step functions (`preprocess_step.py`, `finetune_step.py`, `quantitative_eval_step.py`, `qualitative_eval_step.py`, `model_registration_step.py`, `deploy_step.py`, `pipeline_utils.py`). |
| `scripts/train.py`, `scripts/requirements.txt` | Fine-tuning entry point and its training-container dependencies. |
| `eval/requirements.txt` | Dependencies for the quantitative-evaluation step. |
| `args.yaml`, `config.yaml` | Training hyperparameters and the `@remote` / pipeline runtime config. |

The companion workshop pages live at
[`/generative-ai/mlops/`](https://catalog.workshops.aws/) — see the workshop
content repo for the prose framing.

## MLflow on SageMaker AI

This lab follows the Immersion Day convention and uses a **SageMaker Managed
MLflow App** named `mlflow-app`. The setup cell in each notebook auto-discovers
that app via `list_mlflow_apps()` and creates one (with
`ModelRegistrationMode=AutoModelRegistrationEnabled`) if none exists — the same
pattern used in the
[`classical-ml/lab-5-mlops/mlflow-tracking`](../../classical-ml/lab-5-mlops/mlflow-tracking)
lab. The resolved app ARN is kept in the variable `mlflow_tracking_server_arn`,
which the `steps/` modules consume unchanged.

## Provenance

These notebooks originated from the
[`fine-tuning-with-sagemakerai-and-bedrock`](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/tree/main/workshops/fine-tuning-with-sagemakerai-and-bedrock/task_05_fmops)
workshop (`task_05_fmops`) in `aws-samples/generative-ai-on-amazon-sagemaker`.
They were pulled into this repo so the notebook surface matches the workshop
layout. Changes made on import to match Immersion Day conventions:

- MLflow resolution switched from the older MLflow **Tracking Server** API
  (`describe_mlflow_tracking_server`, name `genai-mlflow-tracker`) to the
  MLflow **Apps** API (`list_mlflow_apps`, name `mlflow-app`, auto-create).
- `sagemaker` / `sagemaker-mlflow` pins aligned to the Immersion Day baseline
  (`sagemaker>3,<4`, `sagemaker-mlflow>=0.5.0,<1`).
- Prerequisite wording updated from "MLflow tracking server" to "MLflow app".

> **Not yet validated end-to-end in this repo.** The training/eval container
> pins (`transformers`, `peft`, `trl`, `mlflow`, etc.) are unchanged from
> upstream and have not been re-run against the Immersion Day SageMaker
> Distribution image. Run both notebooks through once before using in a live
> workshop.

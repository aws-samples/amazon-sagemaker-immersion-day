# Temporary preprocess step (to be changed with new dataset)
import boto3
from sagemaker.s3_utils import parse_s3_url
import sagemaker
from sagemaker.s3 import S3Uploader
from datasets import load_dataset
from datasets import Dataset
import pandas as pd
from steps.utils import safe_open_w, write_to_file
import json
import os

def preprocess(output_data_path):

    # Download data from sciq dataset ( https://huggingface.co/datasets/allenai/sciq )
    dataset_name = 'sciq'
    dataset = load_dataset(dataset_name)

    dataset_training_df = pd.DataFrame(dataset['train'])
    dataset_validation_df = pd.DataFrame(dataset['test'])

    dataset_training_df = dataset_training_df.sample(n=5000, random_state=42, ignore_index=True)

    # Create DAFT dataset
    data_train_daft = " \n".join(((dataset_training_df.drop_duplicates(subset=['support']))['support']))
    write_to_file(data_train_daft, f"./{dataset_name}/dataset_finetune_daft.txt")

    # Create IST dataset
    include_context = False

    if include_context:
        fields = ['support', 'question', 'correct_answer']
        template = {
            "prompt": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Input:\n{support}",
            "completion": "{correct_answer}"
        }

    else:
        fields = ['question', 'correct_answer']
        template = {
            "prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: Answer this question:\n{question}\n",
            "completion": "{correct_answer}"
        }

    dataset_train_ist_df = dataset_training_df[fields].copy()
    dataset_fine_tune_ist = Dataset.from_pandas(dataset_train_ist_df)
    dataset_fine_tune_ist.to_json(f"./{dataset_name}/dataset_finetune_ist.jsonl", orient='records', lines=True)

    # Create evaluation dataset
    with safe_open_w(f"./{dataset_name}/template.json") as text_file:
        json.dump(template, text_file)

    # Print evaluation dataset

    if include_context:
        dataset_validation_with_context_df = dataset_validation_df.copy()
        dataset_validation_with_context_df[
            "model_input"] = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n" + \
                             dataset_validation_with_context_df["question"] + "\n\n### Input:\n" + \
                             dataset_validation_with_context_df["support"]
        dataset_validation_with_context_df = dataset_validation_with_context_df[
            ['model_input', 'correct_answer']].copy()
        dataset_validation_with_context_df = dataset_validation_with_context_df.rename(
            columns={"correct_answer": "target_output"})
        dataset_evaluation = Dataset.from_pandas(dataset_validation_with_context_df)
        print("Evaluation dataset example: ", dataset_evaluation[0])
    else:
        dataset_validation_no_context_df = dataset_validation_df.copy()
        dataset_validation_no_context_df[
            "model_input"] = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: Answer this question:\n" + \
                             dataset_validation_no_context_df["question"]
        dataset_validation_no_context_df = dataset_validation_no_context_df[
            ['model_input', 'correct_answer']].copy()
        dataset_validation_no_context_df = dataset_validation_no_context_df.rename(
            columns={"correct_answer": "target_output"})
        dataset_evaluation = Dataset.from_pandas(dataset_validation_no_context_df)
        print("Evaluation dataset example: ", dataset_evaluation[0])

    print("Export Evaluation dataset: dataset_evaluation.jsonl")
    dataset_evaluation.to_json(f"./{dataset_name}/dataset_evaluation.jsonl")

    # Copy files to S3

    output_bucket = sagemaker.Session().default_bucket()
    output_s3_path = output_bucket + "/datasets"
    data_location = f"s3://{output_s3_path}/" + dataset_name

    print(data_location)

    fine_tune_data_ist_location = f"{data_location}/fine_tuning/instruction_fine_tuning"
    fine_tune_data_daft_location = f"{data_location}/fine_tuning/domain_adaptation_fine_tuning"
    evaluation_data_location = f"{data_location}/evaluation/automatic"

    if (os.path.isfile(f"{dataset_name}/template.json")):
        print("Uploading custom template...")
        S3Uploader.upload(f"{dataset_name}/template.json", fine_tune_data_ist_location)
        print("Done")

    if (os.path.isfile(f"{dataset_name}/dataset_finetune_ist.jsonl")):
        print("Uploading instruction tuning dataset...")
        S3Uploader.upload(f"{dataset_name}/dataset_finetune_ist.jsonl", fine_tune_data_ist_location)
        print(f"Fine-tuning ist data: {fine_tune_data_ist_location}")

    if (os.path.isfile(f"{dataset_name}/dataset_finetune_daft.txt")):
        print("Uploading domain adaptation tuning dataset...")
        S3Uploader.upload(f"{dataset_name}/dataset_finetune_daft.txt", fine_tune_data_daft_location)
        print(f"Fine-tuning daft data: {fine_tune_data_daft_location}")

    if (os.path.isfile(f"{dataset_name}/dataset_evaluation.jsonl")):
        print("Uploading evaluation dataset...")
        S3Uploader.upload(f"{dataset_name}/dataset_evaluation.jsonl", evaluation_data_location)
        print(f"Evaluation data: {evaluation_data_location}")

    print(output_data_path)
    print(fine_tune_data_ist_location)
    print(fine_tune_data_daft_location)
    print(f"{evaluation_data_location}/dataset_evaluation.jsonl")

    return {"output_data_path": output_data_path,
            "fine_tune_data_ist_location": fine_tune_data_ist_location,
            "fine_tune_data_daft_location": fine_tune_data_daft_location,
            "evaluation_data_location": f"{evaluation_data_location}/dataset_evaluation.jsonl"
            }

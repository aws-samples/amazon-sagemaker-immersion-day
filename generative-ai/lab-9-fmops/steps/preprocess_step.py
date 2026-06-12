# **Preprocessing Step**

# This step handles data preparation. We are going to prepare data for training and evaluation. We will log this data in MLflow
import boto3
import shutil
from sagemaker.core.helper.session_helper import Session
import os
import pandas as pd
from sagemaker.core.config import load_sagemaker_config
import mlflow
import traceback
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from random import randint
from sagemaker.mlops.workflow.function_step import step
from .pipeline_utils import (
    PIPELINE_INSTANCE_TYPE,
    # template_dataset,
    SYSTEM_PROMPT,
    convert_to_messages
)


@step(
    name="DataPreprocessing",
    instance_type=PIPELINE_INSTANCE_TYPE,
    display_name="Data Preprocessing",
    keep_alive_period_in_seconds=900
)
def preprocess(
    tracking_server_arn: str,
    input_path: str,
    experiment_name: str,
    run_name: str,
) -> tuple:
    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)

    # Preprocessing code
    try:
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            with mlflow.start_run(run_name="Processing", nested=True):
                # Initialize SageMaker and S3 clients
                sagemaker_session = Session()
                s3_client = boto3.client('s3')

                bucket_name = sagemaker_session.default_bucket()
                default_prefix = sagemaker_session.default_bucket_prefix
                configs = load_sagemaker_config()

                # Set paths
                if default_prefix:
                    input_path = f'{default_prefix}/datasets/llm-fine-tuning-modeltrainer-sft'
                else:
                    input_path = f'datasets/llm-fine-tuning-modeltrainer-sft'

                # Load dataset with proper error handling
                num_samples = 100
                try:
                    full_dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split=f"train[:{num_samples}]")
                except Exception as e:
                    error_msg = f"Error loading dataset: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg)
                    raise RuntimeError(f"Failed to load dataset: {str(e)}")

                # Split dataset
                train_test_split_datasets = full_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
                train_dataset = train_test_split_datasets["train"]
                test_dataset = train_test_split_datasets["test"]
                print(f"Number of train elements: {len(train_dataset)}")
                print(f"Number of test elements: {len(test_dataset)}")

                train_dataset = train_dataset.map(convert_to_messages, remove_columns=list(full_dataset.features), fn_kwargs={"system_prompt": SYSTEM_PROMPT})
                test_dataset = test_dataset.map(convert_to_messages, remove_columns=list(full_dataset.features), fn_kwargs={"system_prompt": SYSTEM_PROMPT})
                #grab a sample from the training and test sets
                print(f"Train Sample:\n{train_dataset[randint(0, len(train_dataset)-1)]}\n\n")
                print(f"Test Sample:\n{test_dataset[randint(0, len(test_dataset)-1)]}\n\n")

                # Log dataset statistics if MLflow is enabled
                mlflow.log_param("dataset_source", "FreedomIntelligence/medical-o1-reasoning-SFT")
                mlflow.log_param("train_size", len(train_dataset))
                mlflow.log_param("test_size", len(test_dataset))
                mlflow.log_param("dataset_sample_size", num_samples)  # Log that we're using a subset of 100 samples
                # save train_dataset to s3 using our SageMaker session
                if default_prefix:
                    input_path = f'{default_prefix}/datasets/llm-fine-tuning-modeltrainer-sft'
                else:
                    input_path = f'datasets/llm-fine-tuning-modeltrainer-sft'
                
                # Save datasets to s3
                # We will fine tune only with 20 records due to limited compute resource for the workshop
                train_dataset.to_json("./data/train/dataset.json", orient="records")
                test_dataset.to_json("./data/test/dataset.json", orient="records")
                
                s3_client.upload_file("./data/train/dataset.json", bucket_name, f"{input_path}/train/dataset.json")
                train_dataset_s3_path = f"s3://{bucket_name}/{input_path}/train/dataset.json"
                s3_client.upload_file("./data/test/dataset.json", bucket_name, f"{input_path}/test/dataset.json")
                test_dataset_s3_path = f"s3://{bucket_name}/{input_path}/test/dataset.json"
                
                shutil.rmtree("./data")
                
                print(f"Training data uploaded to:")
                print(train_dataset_s3_path)
                print(test_dataset_s3_path)
                
                mlflow.log_param("train_data_path", train_dataset_s3_path)
                mlflow.log_param("test_dataset_path", test_dataset_s3_path)

                print(f"Datasets uploaded to:")
                print(train_dataset_s3_path)
                print(test_dataset_s3_path)

    except Exception as e:
        error_msg = f"Critical error in preprocessing: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise RuntimeError(f"Preprocessing failed: {str(e)}")

    return run_id, train_dataset_s3_path, test_dataset_s3_path
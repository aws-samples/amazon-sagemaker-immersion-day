import os
import argparse
from datetime import datetime

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.function_step import step
from sagemaker.workflow.step_outputs import get_step

# Import the necessary steps
from steps.preprocess import preprocess
from steps.evaluation import evaluation
from steps.cleanup import cleanup
from steps.deploy import deploy

from lib.utils import ConfigParser
from lib.utils import find_model_by_name

if __name__ == "__main__":
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = os.getcwd()

    sagemaker_session = sagemaker.session.Session()

    # Define data location either by providing it as an argument or by using the default bucket
    default_bucket = sagemaker.Session().default_bucket()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-input-data-path",
        "--input-data-path",
        dest="input_data_path",
        default=f"s3://{default_bucket}/llm-evaluation-at-scale-example",
        help="The S3 path of the input data",
    )
    parser.add_argument(
        "-config",
        "--config",
        dest="config",
        default="",
        help="The path to .yaml config file",
    )
    args = parser.parse_args()

    # Initialize configuration for data, model, and algorithm
    if args.config:
        config = ConfigParser(args.config).get_config()
    else:
        config = ConfigParser("pipeline_config.yaml").get_config()

    evalaution_exec_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    pipeline_name = config["pipeline"]["name"]
    dataset_config = config["dataset"]  # Get dataset configuration
    input_data_path = args.input_data_path + "/" + dataset_config["input_data_location"]
    output_data_path = (
        args.input_data_path + "/output_" + pipeline_name + "_" + evalaution_exec_id
    )

    print("Data input location:", input_data_path)
    print("Data output location:", output_data_path)

    algorithms_config = config["algorithms"]  # Get algorithms configuration

    model_config = find_model_by_name(config["models"], "llama2-7b")
    model_id = model_config["model_id"]
    model_version = model_config["model_version"]
    evaluation_config = model_config["evaluation_config"]
    endpoint_name = model_config["endpoint_name"]

    model_deploy_config = model_config["deployment_config"]
    deploy_instance_type = model_deploy_config["instance_type"]
    deploy_num_instances = model_deploy_config["num_instances"]

    # Construct the steps
    processed_data_path = step(preprocess, name="preprocess")(
        input_data_path, output_data_path
    )

    endpoint_name = step(deploy, name=f"deploy_{model_id}")(
        model_id,
        model_version,
        endpoint_name,
        deploy_instance_type,
        deploy_num_instances,
    )

    evaluation_results = step(
        evaluation,
        name=f"evaluation_{model_id}",
        keep_alive_period_in_seconds=1200,
        pre_execution_commands=[
            "pip install fmeval==0.2.0",
        ],
    )(
        processed_data_path,
        endpoint_name,
        dataset_config,
        model_config,
        algorithms_config,
        output_data_path,
    )

    last_pipeline_step = evaluation_results

    if model_config["cleanup_endpoint"]:
        cleanup = step(cleanup, name=f"cleanup_{model_id}")(model_id, endpoint_name)
        get_step(cleanup).add_depends_on([evaluation_results])
        last_pipeline_step = cleanup

    # Define the Sagemaker Pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[last_pipeline_step],
    )

    # Build and run the Sagemaker Pipeline
    pipeline.upsert(role_arn=sagemaker.get_execution_role())
    # pipeline.upsert(role_arn="arn:aws:iam::<...>:role/service-role/AmazonSageMaker-ExecutionRole-<...>")

    pipeline.start()

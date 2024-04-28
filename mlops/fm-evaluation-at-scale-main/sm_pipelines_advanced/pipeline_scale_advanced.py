import os
import argparse
from datetime import datetime

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.function_step import step
from sagemaker.workflow.step_outputs import get_step

# Import the necessary steps
from model_providers.steps.preprocess import preprocess
from model_providers.steps.selection import selection
from model_providers.steps.evaluation import evaluation
from model_providers.steps.register import register

from lib.utils import ConfigParser, is_finetuning, create_training_job_name
from lib.import_models import import_models
from lib.utils import get_step_name


if __name__ == "__main__":
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = os.getcwd()

    sagemaker_session = sagemaker.session.Session()

    # Define data location either by providing it as an argument or by using the default bucket
    default_bucket = sagemaker.Session().default_bucket()
    parser = argparse.ArgumentParser()
    parser.add_argument("-input-data-path", "--input-data-path", dest="input_data_path",
        default=f"s3://{default_bucket}/llm-evaluation-at-scale-example", help="The S3 path of the input data")
    parser.add_argument("-config", "--config", dest="config", default="", help="The path to .yaml config file")
    args = parser.parse_args()

    # Initialize configuration for data, model, and algorithm
    if args.config:
        config = ConfigParser(args.config).get_config()
    else:
        #config = ConfigParser('pipeline_config.yaml').get_config()
        #config = ConfigParser('pipeline_finetuning_config.yaml').get_config()
        #config = ConfigParser('pipeline_scale_config.yaml').get_config()
        config = ConfigParser('pipeline_scale_hybrid_config.yaml').get_config()

    evaluation_exec_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    pipeline_name = config["pipeline"]["name"]
    dataset_config = config["dataset"]  # Get dataset configuration
    input_data_path = args.input_data_path + "/" + dataset_config["input_data_location"]
    output_data_path = (
        args.input_data_path + "/output_" + pipeline_name + "_" + evaluation_exec_id
    )

    print("Data input location:", input_data_path)
    print("Data output location:", output_data_path)

    algorithms_config = config["algorithms"] # Get algorithms configuration
    model_registry_config = config["model_registry"]

    # Construct the steps
    preprocess_step_ret = step(preprocess, name="preprocess")(input_data_path, output_data_path)
    # Import the models from config
    models = import_models(config)

    managing_multi_model = True if len(models) > 1 else False

    # Crate finetune, deploy and evaluation step for each model:
    evaluation_results_ret_list = []
    for model in models:

        model.config["output_data_path"] = output_data_path

        if is_finetuning(model):
            model.config["finetuning_config"]["training_job_name"] = create_training_job_name(model)
            model.config["finetuning_config"]["train_data_path"] = args.input_data_path + "/" + model.config["finetuning_config"]["train_data_path"]
            model.config["finetuning_config"]["validation_data_path"] = args.input_data_path + "/" + model.config["finetuning_config"]["validation_data_path"]

            finetune_step_ret = step(model.finetune_step, name=get_step_name("finetune", model), keep_alive_period_in_seconds=2400)(model)
            deploy_step_ret = step(model.deploy_finetuned_step, name=get_step_name("deploy", model))(model, finetune_step_ret)
        else:
            deploy_step_ret = step(model.deploy_step, name=get_step_name("deploy", model))(model)

        evaluation_step_ret = (step(evaluation,
                                    name=get_step_name("evaluation", model),
                                    keep_alive_period_in_seconds=1200,
                                    pre_execution_commands=["pip install fmeval==0.2.0"])
                               (model, dataset_config, algorithms_config, preprocess_step_ret, deploy_step_ret))

        evaluation_results_ret_list.append(evaluation_step_ret)

    selection_step_ret = None
    if managing_multi_model:
        selection_step_ret = (step(selection,
                                   name="model_selection",
                                   pre_execution_commands=["pip install fmeval==0.2.0"])
                              (*evaluation_results_ret_list))

    model_registry_ret = (step(register,
                               name="best_model_registration",
                               pre_execution_commands=["pip install fmeval==0.2.0"])
                          (models, model_registry_config, selection_step_ret, *evaluation_results_ret_list))

    # Create cleanup steps
    pipeline_ret_list = []
    for model in models:
        if model.config["cleanup_endpoint"]:
            cleanup_step_ret = step(model.cleanup_step, name=get_step_name("cleanup", model))(model)
            get_step(cleanup_step_ret).add_depends_on([model_registry_ret])
            pipeline_ret_list.append(cleanup_step_ret)

    if len(pipeline_ret_list) == 0:
        pipeline_ret_list.append(model_registry_ret)

    # Define the Sagemaker Pipeline
    pipeline = Pipeline(name=pipeline_name, steps=pipeline_ret_list)

    # Build and run the Sagemaker Pipeline
    pipeline.upsert(role_arn=sagemaker.get_execution_role())
    #pipeline.upsert(role_arn="arn:aws:iam::<...>:role/service-role/AmazonSageMaker-ExecutionRole-<...>")

    pipeline.start()

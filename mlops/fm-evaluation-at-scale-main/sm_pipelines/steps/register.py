import json

import s3fs as s3fs
import boto3
import sagemaker
from sagemaker import ModelMetrics, MetricsSource
from sagemaker.s3_utils import s3_path_join
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.utils import unique_name_from_base
from sagemaker import image_uris, model_uris, Model, script_uris
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.predictor import Predictor
from sagemaker import get_execution_role
from sagemaker import ModelPackage


def register(
    evaluation,
    model_package_group_name,
    bucket,
):
    sagemaker_session = sagemaker.Session()
    sm_client = boto3.client("sagemaker")

    model_config = evaluation["model_config"]
    eval_output = evaluation["eval_output"]
    eval_result = eval_output[0][0]

    model_id = model_config["model_id"]
    model_version = model_config["model_version"]
    print(model_id, model_version)

    try:
        sm_client.describe_model_package_group(
            ModelPackageGroupName=model_package_group_name
        )
    except:
        model_package_group_input_dict = {
            "ModelPackageGroupName": model_package_group_name,
            "ModelPackageGroupDescription": "Model Package group for FM models",
        }
        create_model_package_group_response = sm_client.create_model_package_group(
            **model_package_group_input_dict
        )
        print(
            "ModelPackageGroup Arn : {}".format(
                create_model_package_group_response["ModelPackageGroupArn"]
            )
        )
    best_model = JumpStartModel(
        model_id=model_id,
        model_version=model_version,
    )

    if (isinstance(best_model.model_data, dict) and 'S3DataSource' in best_model.model_data):
        best_model.model_data = best_model.model_data['S3DataSource']['S3Uri']

    model_package = best_model.register(
        model_package_group_name=model_package_group_name,
        content_types=["application/json"],
        response_types=["application/json"],
        customer_metadata_properties={
            "score": str(eval_result.dataset_scores[0].value),
            "algorithm": eval_result.dataset_scores[0].name,
        },
        skip_model_validation="All",
    )
    model_package_arn = model_package.model_package_arn
    return model_package_arn

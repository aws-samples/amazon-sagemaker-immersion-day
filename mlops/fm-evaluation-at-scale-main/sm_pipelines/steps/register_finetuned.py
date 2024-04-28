import json
import os

import s3fs as s3fs
import boto3
import sagemaker
from sagemaker.jumpstart.estimator import JumpStartEstimator
from sagemaker import get_execution_role
from sagemaker.utils import unique_name_from_base
from sagemaker import ModelMetrics, MetricsSource
from sagemaker.s3_utils import s3_path_join


def register_finetuned(
    training_job_name, model_id, model_package_group_name, evaluation_result, bucket
):
    sagemaker_session = sagemaker.Session()
    #  role = get_execution_role()
    sm_client = boto3.client("sagemaker")

    eval_output = evaluation_result["eval_output"]
    eval_result = eval_output[0][0]

    # Upload evaluation report to s3
    eval_file_name = unique_name_from_base("evaluation")
    eval_report_s3_uri = s3_path_join(
        "s3://",
        bucket,
        model_package_group_name,
        f"evaluation-report/{eval_file_name}.json",
    )
    s3_fs = s3fs.S3FileSystem()
    eval_report_str = json.dumps(
        {
            "score": eval_result.dataset_scores[0].value,
            "algorithm": eval_result.dataset_scores[0].name,
        }
    )
    with s3_fs.open(eval_report_s3_uri, "wb") as file:
        file.write(eval_report_str.encode("utf-8"))

    # Create model_metrics as per evaluation report in s3
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=eval_report_s3_uri,
            content_type="application/json",
        )
    )

    try:
        sm_client.describe_model_package_group(
            ModelPackageGroupName=model_package_group_name
        )
    except:
        model_package_group_input_dict = {
            "ModelPackageGroupName": model_package_group_name,
            "ModelPackageGroupDescription": "Sample model package group",
        }
        create_model_package_group_response = sm_client.create_model_package_group(
            **model_package_group_input_dict
        )
        print(
            "ModelPackageGroup Arn : {}".format(
                create_model_package_group_response["ModelPackageGroupArn"]
            )
        )

    estimator = JumpStartEstimator.attach(training_job_name, model_id=model_id)
    model_package = estimator.register(
        model_package_group_name=model_package_group_name,
        content_types=["application/json"],
        response_types=["application/json"],
        customer_metadata_properties={
            "score": str(eval_result.dataset_scores[0].value),
            "algorithm": eval_result.dataset_scores[0].name,
        },
        model_metrics=model_metrics,
    )
    model_package_arn = model_package.model_package_arn
    return model_package_arn

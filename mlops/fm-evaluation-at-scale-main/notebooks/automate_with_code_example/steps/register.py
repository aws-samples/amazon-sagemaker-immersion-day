from sagemaker import ModelMetrics, MetricsSource
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.jumpstart.estimator import JumpStartEstimator
from sagemaker.s3_utils import parse_s3_url
import boto3
import json


def register(model_list, output_data_path, model_package_group_name, model_package_group_description, selection_step_ret, *args):

    # Search best model:
    best_model = None
    best_model_name = selection_step_ret["best_model_name"]
    for model in model_list:
        if model["model_name"] == best_model_name:
            best_model = model
            break

    # Get evaluation report
    eval_output = None
    for results in args:
        if results["model_name"] == best_model_name:
            eval_output = results["evaluation_output"]
            break

    sm_client = boto3.client("sagemaker")
    s3_client = boto3.client("s3")
    eval_result = eval_output[0][0]

     # Upload evaluation report of the best model to s3
    eval_report_s3_uri = output_data_path + "/evaluation-report/" + model["model_name"] + ".json"
    bucket, object_key = parse_s3_url(eval_report_s3_uri)
    eval_report_str = json.dumps(
    {
        "score": eval_result.dataset_scores[0].value,
        "algorithm": eval_result.dataset_scores[0].name,
    })
    
    s3_client.put_object(Body=eval_report_str, Bucket=bucket, Key=object_key)

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
            "ModelPackageGroupDescription": model_package_group_description,
        }
        create_model_package_group_response = sm_client.create_model_package_group(
            **model_package_group_input_dict
        )
        print(
            "ModelPackageGroup Arn : {}".format(
                create_model_package_group_response["ModelPackageGroupArn"]
            )
        )
        
    # Register Model
    model_id = best_model["model_id"]
    model_version = best_model["model_version"]

    if "is_finetuned_model" in model:
        training_job_name = best_model["training_job_name"]

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
    else:
        js_model = JumpStartModel(model_id=model_id, model_version=model_version)

        if isinstance(js_model.model_data, dict) and 'S3DataSource' in js_model.model_data:
            js_model.model_data = js_model.model_data['S3DataSource']['S3Uri']

        model_package = js_model.register(
            model_package_group_name=model_package_group_name,
            image_uri=js_model.image_uri,
            content_types=["application/json"],
            response_types=["application/json"],
            customer_metadata_properties={
                "score": str(eval_result.dataset_scores[0].value),
                "algorithm": eval_result.dataset_scores[0].name,
            },
            skip_model_validation="All",
            model_metrics=model_metrics,
        )

    model_package_arn = model_package.model_package_arn

    return {"model_package_arn", model_package_arn}

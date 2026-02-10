import json
import os
import tempfile
import tarfile

import numpy as np
import s3fs as s3fs
from sagemaker.core.model_metrics import ModelMetrics, MetricsSource
from sagemaker.core.s3.utils import s3_path_join
from sagemaker.core.common_utils import unique_name_from_base
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.model_registry import create_model_package_from_containers

import mlflow

def register(
    model,
    evaluation,
    model_approval_status,
    model_package_group_name,
    bucket,
    experiment_name="sm-pipeline-experiment",
    run_id=None
):
    sagemaker_session = Session()
    region = sagemaker_session.boto_region_name

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_id=run_id) as run:
        with mlflow.start_run(run_name="Register", nested=True) as nested_run:
            # Upload evaluation report to s3
            eval_file_name = unique_name_from_base("evaluation")
            eval_report_s3_uri = s3_path_join(
                "s3://", bucket, f"evaluation-report/{eval_file_name}.json"
            )

            mlflow.log_param('eval_report_s3_uri', eval_report_s3_uri)
            s3_fs = s3fs.S3FileSystem()
            eval_report_str = json.dumps(evaluation)
            with s3_fs.open(eval_report_s3_uri, "wb") as file:
                file.write(eval_report_str.encode("utf-8"))

            model_metrics = ModelMetrics(
                model_statistics=MetricsSource(
                    s3_uri=eval_report_s3_uri,
                    content_type="application/json",
                )
            )

            # 1. Log model to MLflow
            model_info = mlflow.xgboost.log_model(model, artifact_path="model")

            # 2. Save native XGBoost model and create model.tar.gz for SageMaker
            tmp_dir = tempfile.mkdtemp()
            native_model_path = os.path.join(tmp_dir, "xgboost-model")
            model.save_model(native_model_path)

            model_tar_path = tempfile.mktemp(suffix=".tar.gz")
            with tarfile.open(model_tar_path, "w:gz") as tar:
                tar.add(native_model_path, arcname="xgboost-model")

            model_s3_uri = s3_path_join("s3://", bucket, f"models/{unique_name_from_base('model')}/model.tar.gz")
            with s3_fs.open(model_s3_uri, "wb") as f:
                with open(model_tar_path, "rb") as local_f:
                    f.write(local_f.read())

            # 3. Register model package directly via core API (no sagemaker.serve dependency)
            image_uri = image_uris.retrieve(
                framework="xgboost",
                region=region,
                version="3.0-5",
            )
            container_def = {
                "Image": image_uri,
                "ModelDataUrl": model_s3_uri,
            }

            response = create_model_package_from_containers(
                sagemaker_session=sagemaker_session,
                containers=[container_def],
                content_types=["text/csv"],
                response_types=["text/csv"],
                inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
                transform_instances=["ml.m5.xlarge"],
                model_package_group_name=model_package_group_name,
                approval_status=model_approval_status,
                model_metrics=model_metrics._to_request_dict(),
            )
            model_package_arn = response.get("ModelPackageArn")

            mlflow.set_tags({
                'mlflow.source.name': "register.py",
                'mlflow.source.type': 'REGISTER',
            })

            mlflow.log_param('mlflow_model_uri', model_info.model_uri)
            mlflow.log_param('model_package_arn', model_package_arn)
            print(f"Registered Model Package ARN: {model_package_arn}")

    return model_package_arn

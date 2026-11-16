"""Create or update the SageMaker pipeline of the project and start an execution.

CodeBuild runs this script on every commit to the main branch of the model build
repository (see codebuild-buildspec.yml). You can also run it from a notebook or a
terminal with the same arguments.

The pipeline has four @step functions from the steps/ folder:
preprocess -> train -> evaluate -> register. The register step adds a model version
with the status PendingManualApproval to the project's model package group. Approving
that version starts the project's deployment pipeline.
"""
import argparse
import os

import boto3
import mlflow
from botocore.exceptions import ClientError
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.mlops.workflow import Pipeline
from sagemaker.mlops.workflow.function_step import step

from steps.evaluation import evaluate
from steps.preprocess import preprocess
from steps.register import register
from steps.train import train

# Defaults of all @step functions: container image, role, instance type, environment,
# dependencies, and which local files to upload with the step code
CONFIG_TEMPLATE = """SchemaVersion: '1.0'
SageMaker:
  PythonSDK:
    Modules:
      TelemetryOptOut: true
      RemoteFunction:
        ImageUri: '{image_uri}'
        RoleArn: '{role_arn}'
        InstanceType: {instance_type}
        Dependencies: ./requirements.txt
        IncludeLocalWorkDir: true
        EnvironmentVariables: {{'MLFLOW_TRACKING_URI': '{mlflow_tracking_uri}', 'MLFLOW_EXPERIMENT_NAME': '{mlflow_experiment_name}', 'AWS_REGION': '{region}'}}
        CustomFileFilter:
          IgnoreNamePatterns:
            - ".git"
            - "*.ipynb"
            - "*.md"
            - "__pycache__"
"""


# Keep the step instances warm between steps (SageMaker managed warm pools), so that
# the next step doesn't wait for a new instance
KEEP_ALIVE_SECONDS = 600


def project_tags(project_name, project_id):
    """Tags that link a resource to the SageMaker project in the Studio UI."""
    return [
        {"Key": "sagemaker:project-name", "Value": project_name},
        {"Key": "sagemaker:project-id", "Value": project_id},
    ]


def ensure_model_package_group(name, tags):
    """Create the model package group with the project tags if it doesn't exist."""
    sm = boto3.client("sagemaker")
    try:
        sm.describe_model_package_group(ModelPackageGroupName=name)
        print(f"Model package group {name} exists")
    except ClientError:
        sm.create_model_package_group(
            ModelPackageGroupName=name,
            ModelPackageGroupDescription="Model versions registered by the model build pipeline of the project",
            Tags=tags,
        )
        print(f"Created model package group {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-name", required=True)
    parser.add_argument("--role-arn", required=True, help="Role of the pipeline and its steps")
    parser.add_argument("--model-package-group-name", required=True)
    parser.add_argument("--mlflow-tracking-uri", required=True, help="ARN of the SageMaker managed MLflow app")
    parser.add_argument("--mlflow-experiment-name", required=True)
    parser.add_argument("--image-uri", required=True, help="Container image of the pipeline steps")
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--instance-type", default="ml.r5.xlarge")
    args = parser.parse_args()

    tags = project_tags(args.project_name, args.project_id)
    region = boto3.Session().region_name

    # Write the @step defaults and point the SDK to them. The SDK reads the file
    # when it creates a session, so this must happen before Session().
    with open("config.yaml", "w") as f:
        f.write(CONFIG_TEMPLATE.format(
            image_uri=args.image_uri,
            role_arn=args.role_arn,
            instance_type=args.instance_type,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment_name=args.mlflow_experiment_name,
            region=region,
        ))
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = os.getcwd()

    sagemaker_session = Session()
    bucket = sagemaker_session.default_bucket()

    input_path = (
        f"s3://sagemaker-example-files-prod-{region}/datasets/tabular/uci_abalone/abalone.csv"
    )

    ensure_model_package_group(args.model_package_group_name, tags)
    model_approval_status_param = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    # One MLflow parent run per pipeline version. Each step logs a nested run to it.
    with mlflow.start_run(run_name=args.pipeline_name) as run:
        run_id = run.info.run_id
        mlflow.set_tags({"sagemaker:project-name": args.project_name, "codebuild:build-id": os.environ.get("CODEBUILD_BUILD_ID", "local")})

    data = step(preprocess, name="Abalone_Data_Preprocessing", keep_alive_period_in_seconds=KEEP_ALIVE_SECONDS)(
        input_path,
        experiment_name=args.mlflow_experiment_name,
        run_id=run_id,
    )

    model = step(train, name="Model_Training", keep_alive_period_in_seconds=KEEP_ALIVE_SECONDS)(
        train_df=data[0],
        validation_df=data[1],
        experiment_name=args.mlflow_experiment_name,
        run_id=run_id,
    )

    evaluation_result = step(evaluate, name="Model_Evaluation", keep_alive_period_in_seconds=KEEP_ALIVE_SECONDS)(
        model=model,
        test_df=data[2],
        experiment_name=args.mlflow_experiment_name,
        run_id=run_id,
    )

    model_register = step(register, name="Model_Registration", keep_alive_period_in_seconds=KEEP_ALIVE_SECONDS)(
        model=model,
        evaluation=evaluation_result,
        model_approval_status=model_approval_status_param,
        model_package_group_name=args.model_package_group_name,
        bucket=bucket,
        experiment_name=args.mlflow_experiment_name,
        run_id=run_id,
    )

    pipeline = Pipeline(
        name=args.pipeline_name,
        parameters=[model_approval_status_param],
        steps=[model_register],
        sagemaker_session=sagemaker_session,
    )

    pipeline.upsert(role_arn=args.role_arn, description="Model build pipeline of the SageMaker project", tags=tags)
    execution = pipeline.start()
    print(f"Started pipeline execution {execution.arn}")


if __name__ == "__main__":
    main()

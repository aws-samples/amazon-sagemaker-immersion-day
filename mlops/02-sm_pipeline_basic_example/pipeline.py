import os
import argparse

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.mlops.workflow import Pipeline
from sagemaker.mlops.workflow.function_step import step
from sagemaker.core.workflow.parameters import ParameterString

from steps.preprocess import preprocess
from steps.train import train
from steps.evaluation import evaluate
from steps.register import register

import mlflow

if __name__ == "__main__":
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mlflow_tracking_uri', help='MLflow tracking server URI')
    parser.add_argument('--mlflow_experiment_name', help='mlflow_experiment_name')
    parser.add_argument('--sagemaker_pipeline_name', help='name of the SageMaker Pipeline', default="abalone-sm-pipeline-new-sdk")
    args = parser.parse_args()

    sagemaker_session = Session()

    bucket = sagemaker_session.default_bucket()
    input_path = (f"s3://sagemaker-example-files-prod-{sagemaker_session.boto_region_name}/datasets"
                  f"/tabular/uci_abalone/abalone.csv")

    model_pkg_group_name = "abalone-model-new-sdk"
    model_approval_status_param = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run(run_name=args.sagemaker_pipeline_name) as run:
        run_id = run.info.run_id
        print(run)

        data = step(preprocess, name="Abalone_Data_Preprocessing")(
            input_path,
            run_id=run_id
        )

        model = step(train, name="Model_Training")(
            train_df=data[0],
            validation_df=data[1],
            run_id=run_id
        )

        evaluation_result = step(evaluate, name="Model_Evaluation")(
            model=model,
            test_df=data[2],
            run_id=run_id
        )

        model_register = step(register, name="Model_Registration")(
            model=model,
            evaluation=evaluation_result,
            model_approval_status=model_approval_status_param,
            model_package_group_name=model_pkg_group_name,
            bucket=bucket,
            run_id=run_id
        )

        pipeline = Pipeline(
            name=args.sagemaker_pipeline_name,
            parameters=[model_approval_status_param],
            steps=[model_register],
        )

        pipeline.upsert(
            role_arn=get_execution_role()
        )
        pipeline.start()
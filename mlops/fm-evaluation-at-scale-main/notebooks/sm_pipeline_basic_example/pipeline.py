import os

import sagemaker
from sagemaker.utils import unique_name_from_base
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.function_step import step
from sagemaker.workflow.parameters import (ParameterString,)

from steps.preprocess import preprocess
from steps.train import train
from steps.evaluation import evaluate
from steps.register import register


if __name__ == "__main__":
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = os.getcwd()

    sagemaker_session = sagemaker.session.Session()

    # Define input data and default bucket
    bucket=sagemaker_session.default_bucket()
    input_path = (f"s3://sagemaker-example-files-prod-{sagemaker_session.boto_region_name}/datasets"
                  f"/tabular/uci_abalone/abalone.csv")
    
    # Define the name of the model package group to host all the model versions in pending approval state
    model_pkg_group_name = "abalone-model-new-sdk"
    model_approval_status_param = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")

    # Define the steps of the SageMaker Pipeline, i.e. preprocess, train, evaluate, register
    data = step(preprocess, name="Abalone_Data_Preprocessing")(input_path)

    model = step(train, name="Model_Training")(train_df=data[0], validation_df=data[1])

    evaluation_result = step(evaluate, name="Model_Evaluation")(model=model, test_df=data[2])

    model_register = step(register, name="Model_Registration")(
        model=model,
        evaluation=evaluation_result,
        model_approval_status=model_approval_status_param,
        model_package_group_name=model_pkg_group_name,
        bucket=bucket,
    )
    
    # Create the SageMaker Pipeline including name, parameters, and the output of the last step
    pipeline = Pipeline(
        name="abalone-sm-pipeline-new-sdk",
        parameters=[model_approval_status_param],
        steps=[model_register],
    )
    
    # Deploy and start a SageMaker Pipeline execution
    pipeline.upsert(role_arn=sagemaker.get_execution_role())
    pipeline.start()
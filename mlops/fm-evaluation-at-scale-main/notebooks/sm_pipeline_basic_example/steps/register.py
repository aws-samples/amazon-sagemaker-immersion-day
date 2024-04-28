import json

import numpy as np
import s3fs as s3fs
from sagemaker import ModelMetrics, MetricsSource
from sagemaker.s3_utils import s3_path_join
from sagemaker.serve import SchemaBuilder
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.utils import unique_name_from_base


def register(
    model, evaluation, model_approval_status, model_package_group_name, bucket
):
    # Upload evaluation report to s3
    eval_file_name = unique_name_from_base("evaluation")
    eval_report_s3_uri = s3_path_join(
        "s3://", bucket, f"evaluation-report/{eval_file_name}.json"
    )
    s3_fs = s3fs.S3FileSystem()
    eval_report_str = json.dumps(evaluation)
    with s3_fs.open(eval_report_s3_uri, "wb") as file:
        file.write(eval_report_str.encode("utf-8"))

    # Create model_metrics as per evaluation report in s3
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=eval_report_s3_uri,
            content_type="application/json",
        )
    )

    # Build the trained model and register it
    schema_builder = SchemaBuilder(
        sample_input=np.array(["M", 0.455, 0.365, 0.095, 0.514, 0.2245, 0.101, 0.15]),
        sample_output=np.array([15]),
    )
    model_builder = ModelBuilder(
        model=model,
        schema_builder=schema_builder,
    )
    # Notes: The register method is still under build.
    # * The registered model can not be deployed directly, which will be fixed in the next release.
    # * There will be further improvements on the register method,
    #   such as automatically filling in the content_types and response_types parameters.
    model_package = model_builder.build().register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    print(f"Registered Model Package ARN: {model_package.model_package_arn}")

    return model_package.model_package_arn

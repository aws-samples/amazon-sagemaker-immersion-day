# Origin and adaptation notes

## Project template
[`project-template/build-train-deploy-codecommit.yaml`](project-template/build-train-deploy-codecommit.yaml) is based on the SageMaker-provided Service Catalog product **MLOps template for model building, training, and deployment**, version v1.5 (product `prod-wtkby7nctliqw`, provisioning artifact `pa-f6fiw3uqa7yx2`). SageMaker no longer supports this product for new projects because it uses AWS CodeCommit.

Changes compared to v1.5:

- The seed code of both repositories comes from ZIP files in your S3 bucket (parameters `SeedCodeBucket`, `ModelBuildSeedCodeKey`, `ModelDeploySeedCodeKey`) instead of `s3://sagemaker-servicecatalog-seedcode-<region>/toolchain/`.
- New parameters `MlflowTrackingServerArn` and `PipelineStepImageUri`. The model build CodeBuild project passes them, the model package group name, and the MLflow experiment name to the build as environment variables.
- The model build CodeBuild project uses `BUILD_GENERAL1_MEDIUM`, because it installs the full package set of the pipeline steps.
- Both pipelines use CodePipeline type V2. The artifact bucket blocks public access.
- The project is created from the template in S3 with the `CreateProject` API, not from AWS Service Catalog.

## Model build seed code
[`seed-code/model-build`](seed-code/model-build) replaces the original `model-building-workflow-v1.0.zip` seed, which used the SageMaker Python SDK v2. The step functions in `steps/` are the ones of the [SageMaker Pipelines lab](../sagemaker-pipelines/) (SageMaker Python SDK v3, `@step` decorator, MLflow). `pipeline.py` takes its settings as command line arguments, so that CodeBuild can run it.

## Model deployment seed code
[`seed-code/model-deploy`](seed-code/model-deploy) is the deployment seed code of the [amazon-sagemaker-from-idea-to-production](https://github.com/aws-samples/amazon-sagemaker-from-idea-to-production) workshop (`deployment-seed-code/`), which is based on the SageMaker deployment seed `mpg-deployment-config-v1.1.zip`. Changes for the abalone model:

- `test/payload.csv` contains one preprocessed, 10-feature record of the abalone dataset.
- `test/test.py` requires a finite, positive prediction (the number of rings) instead of a probability.
- `test/buildspec.yml` expects 10 features.

## Diagrams
- [`img/deployment-project-architecture.svg`](img/deployment-project-architecture.svg) is the deployment diagram of notebook 05 of the amazon-sagemaker-from-idea-to-production workshop (`img/deployment-project-architecture-aws.svg`), unchanged.
- [`img/model-build-project-architecture.svg`](img/model-build-project-architecture.svg) follows the same style. [`design/build_diagrams.py`](design/build_diagrams.py) generates it with the AWS Architecture Icons in `design/icons`. Run `python design/build_diagrams.py` from this folder after you change it.

## Model build for a SageMaker project

This repository contains the model build code of a SageMaker project. The project created it from the seed code of the SageMaker Immersion Day lab.

Every commit to the `main` branch starts the model build pipeline in AWS CodePipeline. Its CodeBuild project runs `codebuild-buildspec.yml`, which installs `requirements.txt` and runs `pipeline.py`. The script creates or updates the SageMaker pipeline of the project and starts an execution:

1. `Abalone_Data_Preprocessing`: reads the abalone dataset, scales the numeric features, one-hot encodes `sex`, and splits the data into train, validation, and test sets
2. `Model_Training`: trains an XGBoost regression model that predicts the number of rings
3. `Model_Evaluation`: calculates the mean squared error on the test set
4. `Model_Registration`: registers a model version with the status `PendingManualApproval` in the project's model package group

Each step logs a nested run to the SageMaker managed MLflow app of the project. Approving a model version starts the project's deployment pipeline.

## Repository layout

`codebuild-buildspec.yml`
 - the CodeBuild instructions of the `Build` stage. The CodeBuild project sets the environment variables, such as the pipeline name, the role, the MLflow app ARN, and the container image of the steps.

`pipeline.py`
 - builds the SageMaker pipeline from the step functions with the `@step` decorator, writes the step defaults to `config.yaml`, and starts an execution.

`steps/`
 - the step functions: `preprocess.py`, `train.py`, `evaluation.py`, and `register.py`.

`requirements.txt`
 - the packages that CodeBuild and each pipeline step install.

## Change the model
Edit the step functions, for example the hyperparameters in `steps/train.py`, commit, and push to `main`. The push starts a new build, and the pipeline registers a new model version.

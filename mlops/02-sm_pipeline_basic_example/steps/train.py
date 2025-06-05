import pandas as pd
import os

import xgboost
import mlflow

def train(
    train_df,
    validation_df,
    *,
    num_round=50,
    objective="reg:linear",
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.7,
    use_gpu=False,
    experiment_name = "sm-pipeline-experiment",
    run_id=None
):

    # Enable autologging in MLflow
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_id=run_id) as run:
        with mlflow.start_run(run_name="Train", nested=True):
            # enable mlflow autolog
            mlflow.xgboost.autolog()

            y_train = train_df.iloc[:, 0].to_numpy()
            train_df.drop(train_df.columns[0], axis=1, inplace=True)
            x_train = train_df.to_numpy()
            train_dmatrix = xgboost.DMatrix(x_train, label=y_train)

            y_validation = validation_df.iloc[:, 0].to_numpy()
            validation_df.drop(validation_df.columns[0], axis=1, inplace=True)
            x_validation = validation_df.to_numpy()
            validation_dmatrix = xgboost.DMatrix(x_validation, label=y_validation)

            param = {
                "objective": objective,
                "max_depth": max_depth,
                "eta": eta,
                "gamma": gamma,
                "min_child_weight": min_child_weight,
                "subsample": subsample,
                "tree_method": "gpu_hist"
                if use_gpu
                else "hist",  # Use GPU accelerated algorithm
            }

            mlflow.set_tags(
                {
                    'mlflow.source.name': "train.py",
                    'mlflow.source.type': 'TRAIN',
                }
            )

            evaluation_results = {}  # Store accuracy result
            booster = xgboost.train(
                param,
                train_dmatrix,
                num_round,
                evals=[(train_dmatrix, "train"), (validation_dmatrix, "validation")],
                early_stopping_rounds=5,
                evals_result=evaluation_results,
            )


    return booster

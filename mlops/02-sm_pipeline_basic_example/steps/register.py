import json
import os

import mlflow
from mlflow.models import infer_signature


def register(
    model,
    evaluation,
    model_approval_status,
    model_package_group_name,
    bucket,
    experiment_name="sm-pipeline-experiment",
    run_id=None
):
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_id=run_id) as run:
        with mlflow.start_run(run_name="Register", nested=True) as nested_run:
            # Log evaluation metrics
            for metric_name, metric_value in evaluation.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)
            
            # Log evaluation as artifact
            mlflow.log_dict(evaluation, "evaluation.json")
            
            # Log and register model
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name=model_package_group_name,
            )
            
            mlflow.set_tags({
                'mlflow.source.name': "register.py",
                'mlflow.source.type': 'REGISTER',
                'approval_status': model_approval_status,
            })
            
            model_uri = f"runs:/{nested_run.info.run_id}/model"
            print(f"Registered Model URI: {model_uri}")

    return model_uri
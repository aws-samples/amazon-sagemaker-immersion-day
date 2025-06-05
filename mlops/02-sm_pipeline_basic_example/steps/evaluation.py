import numpy as np
import os
import xgboost
import mlflow

from sklearn.metrics import mean_squared_error


def evaluate(model, test_df, experiment_name="sm-pipeline-experiment", run_id=None):
    
    # Enable autologging in MLflow
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="Evaluate", nested=True):
            mlflow.autolog()
            y_test = test_df.iloc[:, 0].to_numpy()
            test_df.drop(test_df.columns[0], axis=1, inplace=True)
            x_test = test_df.to_numpy()
            predictions = model.predict(xgboost.DMatrix(x_test))

            data = {
                "actual": y_test,
                "predicted": predictions,
                "features": x_test  # Optional context
            }
            
            # Log as a table
            mlflow.log_table(data=data, artifact_file="predictions_table.json")
            
            mse = mean_squared_error(y_test, predictions)
            std = np.std(y_test - predictions)
            report_dict = {
                "regression_metrics": {
                    "mse": {"value": mse, "standard_deviation": std},
                },
            }

            mlflow.set_tags(
                {
                    'mlflow.source.name': "evaluation.py",
                    'mlflow.source.type': 'EVALUATION',
                }
            )
            print(f"evaluation report: {report_dict}")
            mlflow.log_metric("test-mse", mse)
            mlflow.log_metric("test-mse-standard_deviation", std)

    return report_dict
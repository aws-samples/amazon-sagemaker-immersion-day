import numpy as np
import pandas as pd
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import mlflow

# Since we get a headerless CSV file, we specify the column names here.
feature_columns_names = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
label_column = "rings"

feature_columns_dtype = {
    "sex": str,
    "length": np.float64,
    "diameter": np.float64,
    "height": np.float64,
    "whole_weight": np.float64,
    "shucked_weight": np.float64,
    "viscera_weight": np.float64,
    "shell_weight": np.float64,
}
label_column_dtype = {"rings": np.float64}


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def preprocess(raw_data_s3_path: str, experiment_name: str = "sm-pipeline-experiment", run_id: str = None) -> tuple[pd.DataFrame, ...]:
    df = pd.read_csv(
        raw_data_s3_path,
        header=None,
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    )

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])    
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_id=run_id) as run:
        with mlflow.start_run(run_name="DataPreprocessing", nested=True):
            # Enable autologging in MLflow
            mlflow.sklearn.autolog(log_datasets=False)

            dataset = mlflow.data.from_pandas(df, source=raw_data_s3_path)
            
            mlflow.log_input(dataset, context="feature-engineering")
            numeric_features = list(feature_columns_names)
            numeric_features.remove("sex")
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_features = ["sex"]
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            preprocess = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ]
            )

            mlflow.set_tags(
                {
                    'mlflow.source.name': "preprocess.py",
                    'mlflow.source.type': 'PREPROCESS',
                }
            )

            y = df.pop("rings")
            X_pre = preprocess.fit_transform(df)
            y_pre = y.to_numpy().reshape(len(y), 1)

            X = np.concatenate((y_pre, X_pre), axis=1)

            np.random.shuffle(X)
            train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    return pd.DataFrame(train), pd.DataFrame(validation), pd.DataFrame(test)

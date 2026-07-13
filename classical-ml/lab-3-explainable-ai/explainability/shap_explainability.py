"""SageMaker Processing entry point: run SHAP explainability + bias metrics for
the credit-risk model and log the reports to Amazon SageMaker managed MLflow.

This is the managed-infrastructure counterpart (Part 2) of the local
explainability run in the notebook. It is executed by a ``FrameworkProcessor``
job, which packages this directory and pip-installs ``requirements.txt`` into
the sklearn container before running this script.

Inputs (mounted under /opt/ml/processing/input):
  * sklearn_model/model.tar.gz  - fitted sklearn transformer (model.joblib)
  * xgb_model/model.tar.gz      - trained XGBoost booster (xgboost-model)
  * test/test.csv               - raw test features (no target column)
  * train/train.csv             - raw training data (includes target) for the
                                  baseline and the bias-metric ground truth

Outputs (written to /opt/ml/processing/output and logged to MLflow):
  * SHAP global plots, single-instance waterfall, per-instance SHAP CSV,
    bias metrics JSON.

MLflow configuration comes from environment variables set on the job:
  MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, LOGNAME.
"""

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")  # headless backend for the processing container

import pandas as pd

import explainability_lib as elib

INPUT_BASE = "/opt/ml/processing/input"
OUTPUT_BASE = "/opt/ml/processing/output"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facet-name", type=str, default="age")
    parser.add_argument("--facet-threshold", type=float, default=40)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--run-name", type=str, default="explainability-managed-job")
    return parser.parse_args()


def main():
    args = parse_args()

    import mlflow

    # ---- Load model artifacts and build the raw->probability pipeline -------
    sklearn_dir = elib.extract_tar(
        os.path.join(INPUT_BASE, "sklearn_model", "model.tar.gz"),
        "/tmp/sklearn_model",
    )
    xgb_dir = elib.extract_tar(
        os.path.join(INPUT_BASE, "xgb_model", "model.tar.gz"),
        "/tmp/xgb_model",
    )
    predict_proba = elib.load_pipeline(sklearn_dir, xgb_dir)

    # ---- Load data ----------------------------------------------------------
    test_df = pd.read_csv(os.path.join(INPUT_BASE, "test", "test.csv"))
    train_df = pd.read_csv(os.path.join(INPUT_BASE, "train", "train.csv"))

    baseline = elib.compute_baseline(train_df)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # ---- SHAP explainability ------------------------------------------------
    _, shap_values, expected_value = elib.compute_shap_values(
        predict_proba, baseline, test_df, nsamples=args.num_samples
    )
    predictions = predict_proba(test_df)

    plot_paths = elib.save_global_plots(shap_values, test_df, OUTPUT_BASE)

    shap_table = elib.build_shap_table(shap_values, test_df, predictions, expected_value)
    shap_csv = os.path.join(OUTPUT_BASE, "shap_values.csv")
    shap_table.to_csv(shap_csv, index=False)

    # Waterfall for the single most-confident bad-credit prediction.
    min_index = int(shap_table["probability_score"].idxmin())
    waterfall_path = elib.save_waterfall(
        shap_values[min_index],
        expected_value,
        test_df[elib.FEATURE_COLUMNS].iloc[min_index].to_numpy(),
        elib.FEATURE_COLUMNS,
        os.path.join(OUTPUT_BASE, "shap_waterfall_worst_case.png"),
    )

    # ---- Bias metrics (computed on the labelled training data) --------------
    train_predictions = predict_proba(train_df)
    train_pred_labels = (train_predictions > 0.5).astype(int)
    bias_metrics = elib.compute_bias_metrics(
        labels=train_df[elib.LABEL_COLUMN].to_numpy(),
        predictions=train_pred_labels,
        facet_values=train_df[args.facet_name].to_numpy(),
        facet_threshold=args.facet_threshold,
    )
    bias_json = os.path.join(OUTPUT_BASE, "bias_metrics.json")
    with open(bias_json, "w") as f:
        json.dump(bias_metrics, f, indent=2)

    # ---- Log everything to MLflow ------------------------------------------
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tags(
            {
                "mlflow.source.name": "explainability/shap_explainability.py",
                "mlflow.source.type": "JOB",
                "analysis": "shap-explainability-and-bias",
            }
        )
        mlflow.log_params(
            {
                "num_samples": args.num_samples,
                "facet_name": args.facet_name,
                "facet_threshold": args.facet_threshold,
                "baseline": ",".join(str(x) for x in baseline),
                "n_explained_instances": int(len(test_df)),
            }
        )

        # Global feature importance = mean(|SHAP|) per feature.
        mean_abs = pd.DataFrame(
            shap_values, columns=elib.FEATURE_COLUMNS
        ).abs().mean().sort_values(ascending=False)
        for feature, value in mean_abs.items():
            mlflow.log_metric(f"mean_abs_shap_{feature}", float(value))

        mlflow.log_metric("shap_expected_value_logodds", expected_value)
        for name, value in bias_metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                mlflow.log_metric(f"bias_{name}", float(value))

        for path in plot_paths + [waterfall_path, shap_csv, bias_json]:
            mlflow.log_artifact(path, artifact_path="explainability_report")

    print("Explainability + bias reports written to", OUTPUT_BASE)
    print("Bias metrics:", json.dumps(bias_metrics, indent=2))


if __name__ == "__main__":
    main()

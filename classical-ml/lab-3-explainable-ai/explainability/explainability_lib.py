"""Reusable explainability and bias helpers for the credit-risk model.

This module is the single source of truth for the SHAP feature-attribution and
the standardized fairness metrics that replace the discontinued Amazon SageMaker
Clarify service (see
https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-availability-change.html).

It is imported both:
  * locally in the notebook (Part 1 - run the reports on the notebook kernel), and
  * inside the SageMaker managed processing job (Part 2 - same code, run on
    managed infrastructure via ``FrameworkProcessor``).

SHAP is the same engine Clarify used internally, and the bias metrics (CI, DPL,
DPPL, DI) are the published, standardized formulas from the Clarify metric
references, computed here directly with pandas.
"""

import os
import tarfile

import numpy as np
import pandas as pd

# Raw feature columns (the test dataset, i.e. everything except the target).
FEATURE_COLUMNS = [
    "status",
    "duration",
    "credit_history",
    "purpose",
    "amount",
    "savings",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "number_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
]

LABEL_COLUMN = "credit_risk"
# Positive outcome for the target: 1 == "good credit".
POSITIVE_LABEL = 1


# --------------------------------------------------------------------------- #
# Model loading / prediction pipeline
# --------------------------------------------------------------------------- #
def extract_tar(tar_path, dest_dir):
    """Extract a .tar.gz model archive into ``dest_dir`` and return ``dest_dir``."""
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)
    return dest_dir


def _find_file(root, candidates):
    """Return the first existing path under ``root`` matching one of ``candidates``."""
    for name in candidates:
        direct = os.path.join(root, name)
        if os.path.exists(direct):
            return direct
    # Fall back to a recursive search.
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname in candidates:
                return os.path.join(dirpath, fname)
    raise FileNotFoundError(f"None of {candidates} found under {root}")


def load_pipeline(sklearn_model_dir, xgb_model_dir):
    """Load the fitted sklearn transformer + XGBoost booster and return a
    ``predict_proba`` function that maps *raw* feature rows to the probability
    of the positive (good-credit) class.

    This reproduces, locally, exactly what the Clarify shadow endpoint did:
    raw features -> sklearn one-hot transform -> XGBoost -> probability.
    """
    import joblib
    import xgboost

    transformer = joblib.load(_find_file(sklearn_model_dir, ["model.joblib"]))
    booster = xgboost.Booster()
    booster.load_model(_find_file(xgb_model_dir, ["xgboost-model", "model.bin"]))

    def predict_proba(raw):
        if not isinstance(raw, pd.DataFrame):
            raw = pd.DataFrame(np.asarray(raw), columns=FEATURE_COLUMNS)
        features = transformer.transform(raw[FEATURE_COLUMNS])
        return booster.predict(xgboost.DMatrix(features))

    return predict_proba


def compute_baseline(train_df):
    """Build the SHAP baseline as the per-feature mode of the training data.

    The mode is a good choice for the mostly-categorical credit features and
    yields a baseline whose prediction sits on the good-credit side, so that
    bad-credit predictions can be contrasted against it.
    """
    modes = train_df.drop(columns=[LABEL_COLUMN])[FEATURE_COLUMNS].mode().iloc[0]
    return modes.astype("int").tolist()


# --------------------------------------------------------------------------- #
# SHAP explainability
# --------------------------------------------------------------------------- #
def compute_shap_values(predict_proba, baseline_row, explain_df, nsamples=500, link="logit"):
    """Run Kernel SHAP against the mode baseline.

    ``link="logit"`` puts the expected value and SHAP values in log-odds units,
    matching Clarify's ``use_logit=True`` behaviour, so that
    ``sum(shap_values) + E[y]`` equals the model prediction in logit units.

    Returns ``(explainer, shap_values, expected_value)``.
    """
    import shap

    background = pd.DataFrame([baseline_row], columns=FEATURE_COLUMNS)
    explainer = shap.KernelExplainer(predict_proba, background, link=link)
    shap_values = explainer.shap_values(explain_df[FEATURE_COLUMNS], nsamples=nsamples)
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:  # (n, features, outputs) -> single output
        shap_values = shap_values[:, :, 0]
    expected_value = float(np.asarray(explainer.expected_value).ravel()[0])
    return explainer, shap_values, expected_value


def save_global_plots(shap_values, explain_df, out_dir, show=False):
    """Save the global beeswarm summary plot and the mean-|SHAP| bar plot.

    Returns the list of written file paths.
    """
    import matplotlib.pyplot as plt
    import shap

    os.makedirs(out_dir, exist_ok=True)
    paths = []

    beeswarm_path = os.path.join(out_dir, "shap_summary_beeswarm.png")
    shap.summary_plot(shap_values, explain_df[FEATURE_COLUMNS], show=False)
    plt.title("SHAP summary (impact on model output, log-odds)")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    paths.append(beeswarm_path)

    bar_path = os.path.join(out_dir, "shap_feature_importance_bar.png")
    shap.summary_plot(shap_values, explain_df[FEATURE_COLUMNS], plot_type="bar", show=False)
    plt.title("Global feature importance (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    paths.append(bar_path)

    return paths


def _patch_waterfall():
    """Work around a known ``as pl`` alias bug in some shap waterfall builds."""
    import inspect

    import shap.plots._waterfall

    source = inspect.getsource(shap.plots._waterfall)
    if "as pl\n" in source or "as pl " in source:
        exec(source.replace("as pl", "as plt"), shap.plots._waterfall.__dict__)


def save_waterfall(shap_values_row, expected_value, data_row, feature_names, out_path, show=False):
    """Save a single-instance SHAP waterfall plot explaining one prediction."""
    import matplotlib.pyplot as plt
    import shap

    _patch_waterfall()
    explanation = shap.Explanation(
        values=np.asarray(shap_values_row),
        base_values=expected_value,
        data=np.asarray(data_row),
        feature_names=list(feature_names),
    )
    shap.plots.waterfall(explanation, max_display=20, show=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return out_path


def build_shap_table(shap_values, explain_df, predictions, expected_value):
    """Assemble a per-instance table of predictions + SHAP values + raw features."""
    shap_df = pd.DataFrame(
        shap_values, columns=[f"shap_{c}" for c in FEATURE_COLUMNS]
    ).reset_index(drop=True)
    features = explain_df[FEATURE_COLUMNS].reset_index(drop=True)
    proba = pd.Series(np.asarray(predictions).ravel(), name="probability_score").reset_index(
        drop=True
    )
    prediction = (proba > 0.5).astype(int).rename("prediction")
    table = pd.concat([prediction, proba, shap_df, features], axis=1)
    table.attrs["expected_value"] = expected_value
    return table


# --------------------------------------------------------------------------- #
# Standardized bias metrics (pandas re-implementation of Clarify metrics)
# --------------------------------------------------------------------------- #
def compute_bias_metrics(labels, predictions, facet_values, facet_threshold, positive_label=POSITIVE_LABEL):
    """Compute a core set of Clarify pre-training and post-training bias metrics.

    Convention used here (documented so results are reproducible):
      * facet group ``d`` (disadvantaged / of-interest) = rows where the facet
        value is <= ``facet_threshold`` (e.g. age <= 40).
      * group ``a`` (advantaged) = the remaining rows (e.g. age > 40).

    Metrics (see the Clarify metric references for the exact formulas):
      Pre-training:
        * CI  - Class Imbalance
        * DPL - Difference in Positive Proportions in true Labels
      Post-training:
        * DPPL - Difference in Positive Proportions in Predicted Labels
        * DI   - Disparate Impact
    """
    labels = np.asarray(labels).ravel()
    predictions = np.asarray(predictions).ravel()
    facet_values = np.asarray(facet_values).ravel()

    is_d = facet_values <= facet_threshold
    is_a = ~is_d

    n_a = int(is_a.sum())
    n_d = int(is_d.sum())

    def _pos_rate(mask, values):
        n = int(mask.sum())
        if n == 0:
            return float("nan")
        return float((values[mask] == positive_label).sum()) / n

    # Pre-training: from true labels.
    q_a = _pos_rate(is_a, labels)
    q_d = _pos_rate(is_d, labels)
    ci = (n_a - n_d) / (n_a + n_d) if (n_a + n_d) else float("nan")
    dpl = q_a - q_d

    # Post-training: from predicted labels.
    p_a = _pos_rate(is_a, predictions)
    p_d = _pos_rate(is_d, predictions)
    dppl = p_a - p_d
    di = (p_d / p_a) if p_a not in (0.0, float("nan")) else float("nan")

    return {
        "n_advantaged": n_a,
        "n_disadvantaged": n_d,
        "true_positive_rate_advantaged": q_a,
        "true_positive_rate_disadvantaged": q_d,
        "pred_positive_rate_advantaged": p_a,
        "pred_positive_rate_disadvantaged": p_d,
        "CI_class_imbalance": ci,
        "DPL_diff_positive_labels": dpl,
        "DPPL_diff_positive_predicted_labels": dppl,
        "DI_disparate_impact": di,
    }

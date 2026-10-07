"""Train, evaluate and log baseline + advanced models to MLflow. Run: python -m src.train"""
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import CLASSES, ROOT, load_config, load_splits, to_features

# Tree node types we create ourselves; safe to (de)serialise with skops.
TRUSTED = ["sklearn.tree._tree.Tree", "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
           "sklearn.neural_network._stochastic_optimizers.AdamOptimizer"]


def ece(y, proba, bins=15):
    """Expected calibration error (top-label, equal-width bins; Guo et al. 2017)."""
    conf, pred = proba.max(1), proba.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return total


def apply_temperature(proba, t):
    """softmax(log p / T): T>1 softens over-confident probabilities, argmax is unchanged."""
    z = np.log(np.clip(proba, 1e-12, 1)) / t
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def fit_temperature(y, proba):
    """Temperature that minimises validation NLL (Guo et al. 2017)."""
    nll = lambda t: -np.log(apply_temperature(proba, t)[np.arange(len(y)), y] + 1e-12).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 10), method="bounded").x)


def bootstrap_ci(y, pred, n=1000, seed=0):
    """95% percentile CI of macro-F1 over test-set resamples."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n, len(y)))
    vals = [f1_score(y[i], pred[i], average="macro") for i in idx]
    return np.percentile(vals, [2.5, 97.5])


def mcnemar_p(y, pred_a, pred_b):
    """Exact McNemar test on discordant pairs."""
    a_only = int(((pred_a == y) & (pred_b != y)).sum())
    b_only = int(((pred_a != y) & (pred_b == y)).sum())
    return binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0


def metrics(y, proba):
    pred = proba.argmax(1)
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "roc_auc_ovr": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        "ece": ece(y, proba),
    }


def models(seed):
    pca = lambda: PCA(n_components=100, random_state=seed)
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logreg": make_pipeline(StandardScaler(), pca(),
                                LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=seed),
        "hist_gb": make_pipeline(pca(), HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, class_weight="balanced",
            early_stopping=True, random_state=seed)),
        "mlp": make_pipeline(StandardScaler(), pca(), MLPClassifier(
            hidden_layer_sizes=(256, 128), alpha=1e-3, early_stopping=True,
            max_iter=300, random_state=seed)),
    }


def main():
    cfg = load_config()
    out = ROOT / cfg["out_dir"]
    out.mkdir(exist_ok=True)
    splits = load_splits(cfg)
    X = {s: to_features(x) for s, (x, _) in splits.items()}
    y = {s: lab for s, (_, lab) in splits.items()}

    mlflow.set_tracking_uri(cfg["mlflow_uri"])
    mlflow.set_experiment(cfg["experiment"])
    tags = {"data_md5": hashlib.md5((ROOT / cfg["data_path"]).read_bytes()).hexdigest(),
            "sklearn_version": sklearn.__version__, "mlflow_version": mlflow.__version__}
    rows, test_preds = [], {}
    for name, model in models(cfg["seed"]).items():
        with mlflow.start_run(run_name=name, tags=tags) as run:
            t0 = time.time()
            model.fit(X["train"], y["train"])
            fit_s = time.time() - t0
            mlflow.log_params({"model": name, "n_features": X["train"].shape[1], "seed": cfg["seed"],
                               **{k: v for k, v in model.get_params().items()
                                  if isinstance(v, (int, float, str, bool)) and len(str(v)) < 250}})
            row = {"model": name, "run_id": run.info.run_id, "fit_seconds": round(fit_s, 1)}
            for s in ("val", "test"):
                m = metrics(y[s], model.predict_proba(X[s]))
                mlflow.log_metrics({f"{s}_{k}": v for k, v in m.items()})
                row.update({f"{s}_{k}": round(v, 4) for k, v in m.items()})
            mlflow.log_metric("fit_seconds", fit_s)

            # Calibration: temperature fitted on validation only, evaluated on test
            t = fit_temperature(y["val"], model.predict_proba(X["val"]))
            p_test = model.predict_proba(X["test"])
            pred = p_test.argmax(1)
            test_preds[name] = pred
            lo, hi = bootstrap_ci(y["test"], pred)
            extra = {"temperature": t, "test_ece_temp_scaled": ece(y["test"], apply_temperature(p_test, t)),
                     "test_macro_f1_ci_low": lo, "test_macro_f1_ci_high": hi}
            mlflow.log_metrics(extra)
            row.update({k: round(v, 4) for k, v in extra.items()})

            fig, ax = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay.from_predictions(y["test"], model.predict(X["test"]), normalize="true",
                                                    display_labels=[c[:12] for c in CLASSES],
                                                    xticks_rotation=45, ax=ax, values_format=".2f")
            ax.set_title(f"{name} - test confusion (row-normalised)")
            fig.tight_layout()
            cm_path = out / f"confusion_{name}.png"
            fig.savefig(cm_path, dpi=110)
            plt.close(fig)
            mlflow.log_artifact(str(cm_path))
            mlflow.sklearn.log_model(model, name="model", input_example=X["test"][:2],
                                     skops_trusted_types=TRUSTED)  # self-trained
            rows.append(row)
            print(f"{name:14s} val_macroF1={row['val_macro_f1']:.3f} test_macroF1={row['test_macro_f1']:.3f} ({fit_s:.0f}s)")

    df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    df.to_csv(out / "model_comparison.csv", index=False)

    # Comparison chart: headline metrics on test set
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cols = ["test_macro_f1", "test_balanced_acc", "test_accuracy", "test_roc_auc_ovr"]
    df.set_index("model")[cols].plot.bar(ax=ax, rot=0)
    ax.set_ylim(0, 1)
    ax.set_title("DermaMNIST test metrics by model (model selected on val macro-F1)")
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=110)
    plt.close(fig)

    # Register best model (selected on validation, never test)
    best, second = df.iloc[0], df.iloc[1]
    p = mcnemar_p(y["test"], test_preds[best.model], test_preds[second.model])
    (out / "significance.json").write_text(json.dumps(
        {"model_a": best.model, "model_b": second.model, "mcnemar_exact_p": round(p, 4)}, indent=2))
    print(f"McNemar {best.model} vs {second.model}: p={p:.4f}")
    mv = mlflow.register_model(f"runs:/{best.run_id}/model", cfg["registered_model"])
    (out / "best_model.json").write_text(json.dumps(
        {"model": best.model, "run_id": best.run_id, "version": mv.version,
         "temperature": float(best.temperature),
         "uri": f"models:/{cfg['registered_model']}/{mv.version}"}, indent=2))
    print(df.to_string(index=False))
    print(f"Best (val macro-F1): {best.model} -> registered v{mv.version}")


if __name__ == "__main__":
    main()

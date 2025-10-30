import json
import pathlib
import datetime
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, log_loss,
    precision_score, recall_score, confusion_matrix
)

# ---------------- helpers ----------------

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE with uniform probability bins."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        weight = mask.mean()
        ece += abs(acc - conf) * weight
    return float(ece)

def safe_confusion_matrix(y_true, y_pred):
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)

def fail(msg):
    print(msg)
    raise SystemExit(2)

# ------------- main ----------------------

def train_and_eval(ver: str):
    run_dir = pathlib.Path("models") / ver
    cfg_path = run_dir / "train_config.json"
    snap_path = run_dir / "data_snapshot.json"

    if not cfg_path.exists():
        fail(f"missing {cfg_path}")
    if not snap_path.exists():
        fail(f"missing {snap_path} (run split_and_snapshot first)")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    snap = json.loads(snap_path.read_text(encoding="utf-8"))

    feature_file = pathlib.Path(cfg["feature_file"])
    label_col    = cfg["label_column"]
    feat_cols    = cfg["feature_columns"]
    seed         = int(cfg.get("random_seed", 42))
    models_cfg   = cfg["models"]
    ece_bins     = int(cfg["calibration"]["ece_bins"])

    if not feature_file.exists():
        fail(f"feature file not found: {feature_file}")

    # Load canonical features
    df = pd.read_parquet(feature_file)

    # Validate columns
    required = set(feat_cols + [label_col])
    missing = required.difference(df.columns)
    if missing:
        fail(f"missing columns in features: {sorted(missing)}")

    # --- deterministic split first (so imputers/encoders fit on TRAIN only) ---
    total = snap["rows_train"] + snap["rows_val"]
    test_size = snap["rows_val"] / total if total else 0.20

    train_idx, val_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=snap["split_random_state"],
        shuffle=True,
        stratify=df[label_col]
    )

    X_train_raw, y_train = df.loc[train_idx, feat_cols].copy(), df.loc[train_idx, label_col].copy()
    X_val_raw,   y_val   = df.loc[val_idx,   feat_cols].copy(), df.loc[val_idx,   label_col].copy()

    # --- identify feature types
    cat_cols = [c for c in feat_cols if (X_train_raw[c].dtype == "object") or str(X_train_raw[c].dtype).startswith("category")]
    num_cols = [c for c in feat_cols if c not in cat_cols]

    # --- imputers (fit on train only)
    # numeric: median
    numeric_medians = {c: float(X_train_raw[c].median(skipna=True)) if X_train_raw[c].notna().any() else 0.0 for c in num_cols}
    for c in num_cols:
        X_train_raw[c] = X_train_raw[c].fillna(numeric_medians[c])
        X_val_raw[c]   = X_val_raw[c].fillna(numeric_medians[c])
        # ensure plain numeric dtype
        if str(X_train_raw[c].dtype).startswith("Int"):
            X_train_raw[c] = X_train_raw[c].astype("float64")
            X_val_raw[c]   = X_val_raw[c].astype("float64")

    # categorical: constant "__MISSING__" then LabelEncoder (fit on train only)
    CAT_MISSING = "__MISSING__"
    encoders = {}
    for c in cat_cols:
        X_train_raw[c] = X_train_raw[c].astype("string").fillna(CAT_MISSING)
        X_val_raw[c]   = X_val_raw[c].astype("string").fillna(CAT_MISSING)
        le = LabelEncoder()
        le.fit(X_train_raw[c])
        X_train_raw[c] = le.transform(X_train_raw[c])
        X_val_raw[c]   = le.transform(X_val_raw[c])
        encoders[c] = {"classes_": le.classes_.tolist(), "missing_token": CAT_MISSING}

    # assemble final matrices
    X_train, X_val = X_train_raw, X_val_raw

    # safety: ensure no NaNs remain
    if np.isnan(X_train.to_numpy()).any() or np.isnan(X_val.to_numpy()).any():
        fail("NaNs remain after imputation; aborting.")

    results = {}
    best_auc, best_key, best_model = -1.0, None, None

    for key, mcfg in models_cfg.items():
        mtype = mcfg.get("type")
        params = {k: v for k, v in mcfg.items() if k != "type"}

        if mtype == "logistic_regression":
            if "random_state" not in params:
                params["random_state"] = seed
            model = LogisticRegression(**params)

        elif mtype == "random_forest":
            model = RandomForestClassifier(**params)

        else:
            print(f"Skipping unknown model type: {mtype}")
            continue

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_val, y_prob)
        f1  = f1_score(y_val, y_pred, zero_division=0)
        ll  = log_loss(y_val, y_prob, labels=[0, 1])
        ece = expected_calibration_error(y_val, y_prob, n_bins=ece_bins)
        prec= precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        tn, fp, fn, tp = safe_confusion_matrix(y_val, y_pred)

        results[key] = {
            "auc": auc, "f1": f1, "logloss": ll, "ece": ece,
            "precision": prec, "recall": rec,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
        }

        if auc > best_auc:
            best_auc, best_key, best_model = auc, key, model

    # Write metrics.json
    summary = {
        "version": ver,
        "random_seed": seed,
        "generated_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics_val": results,
        "candidate_selection": {
            "criterion": "max_auc",
            "winner_model_key": best_key,
            "winner_auc": best_auc
        }
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save model package (model + encoders + imputers + schema bits)
    package = {
        "model": best_model,
        "encoders": encoders,                  # per-categorical column LabelEncoder classes
        "numeric_medians": numeric_medians,    # per-numeric column median used for imputation
        "feature_columns": feat_cols,
        "categorical_columns": cat_cols,
        "label_column": label_col,
        "version": ver
    }
    joblib.dump(package, run_dir / "model.pkl")

    print(f"trained {len(results)} models, best={best_key} auc={best_auc:.3f}")
    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        fail("usage: python src/train_baselines.py vYYYYMMDD-HHMM")
    _ver = sys.argv[1]
    train_and_eval(_ver)

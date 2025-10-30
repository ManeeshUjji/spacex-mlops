import json, sys, pathlib, datetime
import pandas as pd
from sklearn.model_selection import train_test_split

def fail(msg):
    print(msg)
    sys.exit(2)

def main():
    if len(sys.argv) != 2:
        fail("usage: python src/split_and_snapshot.py vYYYYMMDD-HHMM")
    ver = sys.argv[1]
    run_dir = pathlib.Path('models')/ver
    cfg_path = run_dir/'train_config.json'
    if not cfg_path.exists():
        fail(f"missing {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    feature_file = pathlib.Path(cfg['feature_file'])
    label_col = cfg['label_column']
    feature_cols = cfg['feature_columns']
    seed = int(cfg.get('random_seed', 42))
    test_size = float(cfg['split']['test_size'])

    if not feature_file.exists():
        fail(f"feature file not found: {feature_file}")

    # load
    df = pd.read_parquet(feature_file)

    # validate columns
    required = set(feature_cols + [label_col])
    missing = required.difference(df.columns)
    if missing:
        fail(f"missing columns in features: {sorted(missing)}")

    # label sanity
    if df[label_col].isna().any():
        fail(f"label column '{label_col}' has nulls")
    if not set(df[label_col].unique()).issubset({0,1}):
        fail(f"label column '{label_col}' must be binary 0/1")

    # stratified split
    y = df[label_col]
    train_idx, val_idx = train_test_split(
        df.index, test_size=test_size, random_state=seed, shuffle=True, stratify=y
    )

    # stats
    def pos_rate(idx):
        if len(idx) == 0: return 0.0
        return float(df.loc[idx, label_col].mean())

    rows_total = int(len(df))
    rows_train = int(len(train_idx))
    rows_val   = int(len(val_idx))

    snapshot = {
        "feature_file": str(feature_file).replace("\\","/"),
        "rows_total": rows_total,
        "rows_train": rows_train,
        "rows_val": rows_val,
        "positive_rate_total": float(df[label_col].mean()),
        "positive_rate_train": pos_rate(train_idx),
        "positive_rate_val": pos_rate(val_idx),
        "generated_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split_random_state": seed
    }

    # write snapshot JSON
    out_path = run_dir/'data_snapshot.json'
    out_path.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')

    print(f"Wrote {out_path} ?")
    print(f"Totals: T={rows_total} train={rows_train} val={rows_val}  "
          f"p1: total={snapshot['positive_rate_total']:.3f} "
          f"train={snapshot['positive_rate_train']:.3f} "
          f"val={snapshot['positive_rate_val']:.3f}")

if __name__ == "__main__":
    main()
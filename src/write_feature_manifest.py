import json, pathlib
import pandas as pd

def main(ver):
    run_dir = pathlib.Path('models')/ver
    cfg = json.loads((run_dir/'train_config.json').read_text(encoding='utf-8'))
    feature_file = pathlib.Path(cfg['feature_file'])

    # Phase-1 manifest (source of truth for path + sha256)
    mpath = pathlib.Path('data/manifest.json')
    manifest = json.loads(mpath.read_text(encoding='utf-8'))

    # Be flexible to key names
    mf_feature = manifest.get('feature_file') or manifest.get('features', {}).get('path')
    mf_sha     = manifest.get('sha256_feature') or manifest.get('features', {}).get('sha256')

    # Load parquet to record concrete dtypes/columns count
    df = pd.read_parquet(feature_file)

    # Build output
    out = {
        "feature_file": str(feature_file).replace("\\","/"),
        "sha256_feature": mf_sha,
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns}
    }

    (run_dir/'feature_manifest.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"Wrote {run_dir/'feature_manifest.json'} ?")

if __name__ == "__main__":
    import sys
    if len(sys.argv)!=2:
        sys.exit("usage: python src/write_feature_manifest.py vYYYYMMDD-HHMM")
    main(sys.argv[1])
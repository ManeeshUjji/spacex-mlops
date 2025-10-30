import json, pathlib, shutil, sys

REQUIRED = [
    'model.pkl',
    'metrics.json',
    'train_config.json',
    'feature_manifest.json',
    'data_snapshot.json'
]

def main(ver):
    src_dir = pathlib.Path('models')/ver
    dst_dir = pathlib.Path('registry')/'models'/ver
    if not src_dir.exists():
        sys.exit(f'missing source dir: {src_dir}')
    dst_dir.mkdir(parents=True, exist_ok=True)

    # verify all required exist in source
    missing = [f for f in REQUIRED if not (src_dir/f).exists()]
    if missing:
        sys.exit(f'missing artifacts in {src_dir}: {missing}')

    # copy with metadata
    for f in REQUIRED:
        shutil.copy2(src_dir/f, dst_dir/f)
    print(f"Mirrored to {dst_dir} ?")

if __name__ == '__main__':
    if len(sys.argv)!=2:
        sys.exit('usage: python src/mirror_to_registry.py vYYYYMMDD-HHMM')
    main(sys.argv[1])
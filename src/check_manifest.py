import json, sys, hashlib, pathlib, datetime

MANIFEST_PATH = pathlib.Path('data/manifest.json')
LOG_PATH = pathlib.Path('logs/phase2_guardrails.log')

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    utcnow = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    if not MANIFEST_PATH.exists():
        msg = f"[{utcnow}] ERROR manifest missing: {MANIFEST_PATH}"
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text((LOG_PATH.read_text() if LOG_PATH.exists() else '') + msg + '\n')
        print(msg)
        sys.exit(2)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    feature_file = manifest.get('feature_file') or manifest.get('features', {}).get('path')
    expected_sha = manifest.get('sha256_feature') or manifest.get('features', {}).get('sha256')

    if not feature_file or not expected_sha:
        msg = f"[{utcnow}] ERROR manifest missing keys: feature_file/sha256_feature"
        LOG_PATH.write_text((LOG_PATH.read_text() if LOG_PATH.exists() else '') + msg + '\n')
        print(msg)
        sys.exit(3)

    p = pathlib.Path(feature_file)
    if not p.exists():
        msg = f"[{utcnow}] ERROR feature file missing: {p}"
        LOG_PATH.write_text((LOG_PATH.read_text() if LOG_PATH.exists() else '') + msg + '\n')
        print(msg)
        sys.exit(4)

    got_sha = sha256_file(p)
    if got_sha.lower() != str(expected_sha).lower():
        msg = f"[{utcnow}] HASH MISMATCH for {p.name}: got {got_sha}, expected {expected_sha}"
        LOG_PATH.write_text((LOG_PATH.read_text() if LOG_PATH.exists() else '') + msg + '\n')
        print(msg)
        sys.exit(5)

    print(f"[{utcnow}] OK: manifest + sha256 verified for {p} ({got_sha[:12]}...)")
    sys.exit(0)

if __name__ == '__main__':
    main()
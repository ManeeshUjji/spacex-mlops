import json, pathlib, sys, datetime

POINTER = pathlib.Path('registry/champion/pointer.json')
LOG = pathlib.Path('logs/promotion.log')

def read_metrics(ver):
    mpath = pathlib.Path('registry/models')/ver/'metrics.json'
    tpath = pathlib.Path('registry/models')/ver/'train_config.json'
    if not mpath.exists():
        sys.exit(f'missing metrics for {ver}: {mpath}')
    if not tpath.exists():
        sys.exit(f'missing train_config for {ver}: {tpath}')
    metrics = json.loads(mpath.read_text(encoding='utf-8'))
    cfg = json.loads(tpath.read_text(encoding='utf-8'))
    winner = metrics['candidate_selection']['winner_model_key']
    auc = float(metrics['metrics_val'][winner]['auc'])
    ece = float(metrics['metrics_val'][winner]['ece'])
    f1  = float(metrics['metrics_val'][winner]['f1'])
    ll  = float(metrics['metrics_val'][winner]['logloss'])
    pol = cfg['promotion_policy']
    return {'auc':auc, 'ece':ece, 'f1':f1, 'logloss':ll, 'winner':winner, 'policy':pol}

def log(line):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text((LOG.read_text() if LOG.exists() else '') + line + '\n', encoding='utf-8')

def main(ver):
    now = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')
    cand = read_metrics(ver)

    if not POINTER.exists():
        # Bootstrap: first ever champion
        POINTER.parent.mkdir(parents=True, exist_ok=True)
        pointer = {
            'champion_version': ver,
            'previous_version': None,
            'promoted_at': now,
            'reason': 'bootstrap (first model)',
            'metrics_snapshot': {
                'auc': cand['auc'], 'ece': cand['ece'], 'f1': cand['f1'], 'logloss': cand['logloss']
            }
        }
        POINTER.write_text(json.dumps(pointer, indent=2), encoding='utf-8')
        print(f'PROMOTED (bootstrap): {ver}')
        log(f'[{now}] PROMOTED bootstrap -> {ver}')
        return

    # Compare to current champion
    ptr = json.loads(POINTER.read_text(encoding='utf-8'))
    cur = ptr['champion_version']
    curm = read_metrics(cur)

    delta_auc = cand['auc'] - curm['auc']
    delta_ece = cand['ece'] - curm['ece']

    thr_auc = float(cand['policy']['auc_improve_at_least'])
    thr_ece = float(cand['policy']['max_calibration_worsening'])

    promote = (delta_auc >= thr_auc) and (delta_ece <= thr_ece)

    if promote:
        pointer = {
            'champion_version': ver,
            'previous_version': cur,
            'promoted_at': now,
            'reason': f'AUC +{delta_auc:.3f} >= +{thr_auc:.3f} and ECE +{delta_ece:.3f} <= +{thr_ece:.3f}',
            'metrics_snapshot': {
                'auc': cand['auc'], 'ece': cand['ece'], 'f1': cand['f1'], 'logloss': cand['logloss']
            }
        }
        POINTER.write_text(json.dumps(pointer, indent=2), encoding='utf-8')
        print(f'PROMOTED: {cur} -> {ver} | ?AUC={delta_auc:.3f}, ?ECE={delta_ece:.3f}')
        log(f'[{now}] PROMOTED {cur} -> {ver} | ?AUC={delta_auc:.3f}, ?ECE={delta_ece:.3f}')
    else:
        print(f'NO PROMOTION: keep {cur}. Candidate {ver} had ?AUC={delta_auc:.3f} (need = {thr_auc:.3f}), '
              f'?ECE={delta_ece:.3f} (need = {thr_ece:.3f}).')
        log(f'[{now}] NO PROMOTION (keep {cur}) vs {ver} | ?AUC={delta_auc:.3f} (thr {thr_auc:.3f}), '
            f'?ECE={delta_ece:.3f} (thr {thr_ece:.3f}).')

if __name__ == '__main__':
    if len(sys.argv)!=2:
        sys.exit('usage: python src/promote_if_better.py vYYYYMMDD-HHMM')
    main(sys.argv[1])
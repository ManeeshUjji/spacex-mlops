import json, pathlib, datetime, sys

def main(ver):
    manifest_path = pathlib.Path('data/manifest.json')
    pointer_path  = pathlib.Path('registry/champion/pointer.json')
    cand_metrics  = pathlib.Path('registry/models')/ver/'metrics.json'

    if not manifest_path.exists() or not pointer_path.exists() or not cand_metrics.exists():
        sys.exit('missing manifest, pointer, or metrics')

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    pointer  = json.loads(pointer_path.read_text(encoding='utf-8'))
    cand     = json.loads(cand_metrics.read_text(encoding='utf-8'))

    champ_ver = pointer['champion_version']
    champ_metrics_path = pathlib.Path('registry/models')/champ_ver/'metrics.json'
    champ = json.loads(champ_metrics_path.read_text(encoding='utf-8'))

    cw = cand['candidate_selection']['winner_model_key']
    aw = champ['candidate_selection']['winner_model_key']

    summary = {
        'candidate': {
            'auc': cand['metrics_val'][cw]['auc'],
            'f1': cand['metrics_val'][cw]['f1'],
            'logloss': cand['metrics_val'][cw]['logloss'],
            'ece': cand['metrics_val'][cw]['ece']
        },
        'champion': {
            'auc': champ['metrics_val'][aw]['auc'],
            'f1': champ['metrics_val'][aw]['f1'],
            'logloss': champ['metrics_val'][aw]['logloss'],
            'ece': champ['metrics_val'][aw]['ece']
        }
    }

    manifest.update({
        'last_training_run': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'candidate_version': ver,
        'champion_version': champ_ver,
        'metric_summary': summary
    })

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"Updated {manifest_path} ?")

if __name__ == '__main__':
    if len(sys.argv)!=2:
        sys.exit('usage: python src/update_manifest_posttrain.py vYYYYMMDD-HHMM')
    main(sys.argv[1])
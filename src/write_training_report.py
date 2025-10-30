import json, pathlib, datetime, subprocess, sys

def main(ver):
    run_dir = pathlib.Path('models')/ver
    metrics = json.loads((run_dir/'metrics.json').read_text(encoding='utf-8'))
    snapshot = json.loads((run_dir/'data_snapshot.json').read_text(encoding='utf-8'))
    pointer  = json.loads(pathlib.Path('registry/champion/pointer.json').read_text(encoding='utf-8'))

    git_sha = subprocess.run(['git','rev-parse','--short','HEAD'], capture_output=True, text=True).stdout.strip()

    report_path = pathlib.Path('reports')/f'training_run_{ver}.txt'
    with report_path.open('w', encoding='utf-8') as f:
        f.write(f"SpaceX MLOps � Training Run {ver}\n")
        f.write("="*60 + "\n\n")

        f.write("Data\n")
        f.write(f"  feature_file: {snapshot['feature_file']}\n")
        f.write(f"  rows: total={snapshot['rows_total']}  train={snapshot['rows_train']}  val={snapshot['rows_val']}\n")
        f.write(f"  pos_rate: total={snapshot['positive_rate_total']:.3f}  "
                f"train={snapshot['positive_rate_train']:.3f}  val={snapshot['positive_rate_val']:.3f}\n\n")

        f.write("Validation Metrics\n")
        for k,v in metrics['metrics_val'].items():
            f.write(f"  {k}:\n")
            for mk,mv in v.items():
                if isinstance(mv,dict): continue
                f.write(f"    {mk:<10}: {mv:.4f}\n")
            f.write("\n")

        cand = metrics['candidate_selection']['winner_model_key']
        f.write(f"Candidate model (winner): {cand}\n")
        f.write(f"Champion version after promotion: {pointer['champion_version']}\n\n")

        f.write("Provenance\n")
        f.write(f"  git_commit: {git_sha}\n")
        f.write(f"  run_at_utc: {datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
    print(f"Wrote {report_path} ?")

if __name__ == '__main__':
    if len(sys.argv)!=2:
        sys.exit('usage: python src/write_training_report.py vYYYYMMDD-HHMM')
    main(sys.argv[1])
import joblib
from pathlib import Path

p = Path(r'registry/models/v20251030-0333/model.pkl')  # <-- adjust to your version folder shown in pointer.json
obj = joblib.load(p)
print('TOP TYPE:', type(obj))

for name in ('predict_proba','predict','decision_function','transform'):
    print(f'top.{name}:', hasattr(obj, name))

if isinstance(obj, dict):
    print('DICT KEYS:', list(obj.keys()))
    for k,v in obj.items():
        print(' -', k, '->', type(v))

import shutil, pathlib, sys
ver = sys.argv[1]
src = pathlib.Path(f'models/{ver}/model.pkl')
dst = pathlib.Path(f'registry/models/{ver}/model.pkl')
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dst)
print('Copied to', dst)

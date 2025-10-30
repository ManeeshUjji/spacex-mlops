.PHONY: fetch validate process features manifest all clean

fetch:
	@echo "TODO: python src/fetch_raw.py"

validate:
	@echo "TODO: python src/validate_raw.py"

process:
	@echo "TODO: python src/clean_transform.py"

features:
	@echo "TODO: python src/build_features.py"

manifest:
	@echo "TODO: python src/manifest.py"

all: manifest

clean:
	@powershell -NoProfile -Command "Remove-Item -Recurse -Force data\\processed\\*, data\\features\\*, reports\\*, logs\\* -ErrorAction SilentlyContinue"

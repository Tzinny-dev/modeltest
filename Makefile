PYTHON ?= python3

.PHONY: install test example validate clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test: install
	$(PYTHON) -m pytest tests/ -q

example:
	$(PYTHON) examples/basic.py

# Generate sample artifacts, then validate them via the CLI + YAML suite
validate:
	$(PYTHON) examples/make_artifacts.py
	modeltest validate --suite examples/suite.yaml --model examples/model.pkl \
		--data examples/validation.csv --target target \
		--train-data examples/train.csv --output reports/model-validation.xml

clean:
	rm -rf reports examples/model.pkl examples/train.csv examples/validation.csv
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

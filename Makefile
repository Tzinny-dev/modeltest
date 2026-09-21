PYTHON ?= python3
RUFF ?= ruff

.PHONY: install lint format test example validate precommit docs docs-build docs-check-links docs-deploy clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(RUFF) check modeltest/ tests/

format:
	$(RUFF) format modeltest/ tests/

test: install
	$(PYTHON) -m pytest tests/ -q

precommit:
	pre-commit install

example:
	$(PYTHON) examples/basic.py

# Generate sample artifacts, then validate them via the CLI + YAML suite
validate:
	$(PYTHON) examples/make_artifacts.py
	modeltest validate --suite examples/suite.yaml --model examples/model.pkl \
		--data examples/validation.csv --target target \
		--train-data examples/train.csv --output reports/model-validation.xml

clean:
	rm -rf reports examples/model.pkl examples/train.csv examples/validation.csv site
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Live-reload docs server (http://localhost:8000) — single-version preview.
# For a versioned preview with the switcher, use: mike serve
docs:
	$(PYTHON) -m mkdocs serve

# Strict build (same as CI; fails on broken links via the link checker).
docs-build:
	$(PYTHON) -m pip install -q mkdocs-material "mkdocstrings[python]" mike markdown requests beautifulsoup4
	$(PYTHON) -m mkdocs build --strict --site-dir site

# Verify external + local links in docs/ and README.md (script propio).
# Requiere: markdown, requests, beautifulsoup4 (instalados por docs-build).
docs-check-links:
	$(PYTHON) scripts/check_links.py

# Deploy versioned docs to GitHub Pages with mike (reads version from
# pyproject.toml). Requires git credentials for the push.
VERSION ?= $(shell $(PYTHON) -c "import re, pathlib; print(re.search(r'^version\\s*=\\s*\"([^\"]+)\"', pathlib.Path('pyproject.toml').read_text()).group(1))")
docs-deploy:
	$(PYTHON) -m mike deploy --push --update-aliases $(VERSION) latest

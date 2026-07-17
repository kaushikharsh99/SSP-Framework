.PHONY: install install-dev format lint test clean docs pre-commit-install

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

format:
	ruff format .
	black .
	isort .

lint:
	ruff check .
	mypy ssp_framework

test:
	pytest

pre-commit-install:
	pre-commit install

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +

docs:
	cd docs && make html

# Contributing to SSP Framework

We welcome contributions to the Spectrum-to-Signal Principle (SSP) Framework! As a research-focused repository, maintaining high standards of code quality, reproducibility, and clarity is essential.

## How to Contribute

### 1. Report Bugs or Suggest Features
- Check the existing issue tracker before opening a new issue.
- Use the provided issue templates. Provide as much details as possible, including system configuration, steps to reproduce, or theoretical rationale.

### 2. Code Contributions
- Fork the repository and create your branch from `main`.
- Follow the coding style guidelines. We use `black`, `isort`, and `ruff` for formatting and linting.
- Add unit tests for any new modules or code logic under the `tests/` directory.
- Ensure all tests pass (`make test`) and linting checks succeed (`make lint`).
- If you add a new model or experiment config, update documentation inside the `docs/` folder.

## Setup Development Environment

1. Clone your fork of the repository:
   ```bash
   git clone https://github.com/your-username/SSP-Framework.git
   cd SSP-Framework
   ```

2. Create a virtual environment and install the development dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   make install-dev
   ```

3. Enable pre-commit hooks:
   ```bash
   make pre-commit-install
   ```

## Development Workflow

- Run auto-formatting:
  ```bash
  make format
  ```
- Run static checks and type checking:
  ```bash
  make lint
  ```
- Run test suite:
  ```bash
  make test
  ```

## Pull Request Guidelines

1. Link relevant issues in the PR description.
2. Keep PRs focused on a single logical change or research concept.
3. Write clear commit messages.
4. Include experimental results or validation logs if introducing new algorithms.

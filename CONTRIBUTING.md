# Contributing to FlowState

Thank you for your interest in contributing to FlowState! This document provides guidelines and information for contributors.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/flowstate-io/flowstate.git
cd flowstate

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=flowstate --cov-report=term-missing
```

## Code Quality

```bash
# Linting
ruff check src/ tests/

# Formatting
ruff format src/ tests/

# Type checking
mypy src/flowstate/
```

## Pull Request Process

1. Fork the repository and create your branch from `main`.
2. Add tests for any new functionality.
3. Ensure all tests pass and code quality checks are clean.
4. Update documentation if you're changing public APIs.
5. Submit your pull request with a clear description of the changes.

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `test:` — adding or updating tests
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `perf:` — performance improvement

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

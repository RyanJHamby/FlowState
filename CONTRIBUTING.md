# Contributing to FlowState

## Development Setup

```bash
git clone https://github.com/RyanJHamby/flowstate.git && cd flowstate

# Python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Rust core (requires Rust toolchain + maturin)
cd flowstate-core && maturin develop --release && cd ..
```

## Running Tests

```bash
# Python (636 tests)
python -m pytest tests/ -v

# Python with coverage
python -m pytest tests/ -v --cov=flowstate --cov-report=term-missing

# Rust (132 tests)
cd flowstate-core && cargo test --no-default-features

# Rust benchmarks
cargo bench --no-default-features

# Full-stack Python benchmarks
python benchmarks/bench_full_suite.py
```

## Code Quality

```bash
# Python linting
ruff check src/ tests/

# Python formatting
ruff format src/ tests/

# Python type checking
mypy src/flowstate/ --ignore-missing-imports

# Rust formatting
cd flowstate-core && cargo fmt -- --check

# Rust linting
cargo clippy --no-default-features -- -D warnings
```

## Pull Request Process

1. Create your branch from `main`.
2. Add tests for any new functionality.
3. Ensure all tests pass and code quality checks are clean.
4. Update documentation if you are changing public APIs.
5. Submit your pull request with a clear description of the changes.

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `test:` — adding or updating tests
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `perf:` — performance improvement
- `chore:` — build, CI, or tooling changes

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

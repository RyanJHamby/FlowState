set shell := ["bash", "-c"]

# Python tests
test:
	python -m pytest tests/ -v

# Rust tests (flowstate-core)
test-rust:
	cd flowstate-core && cargo test --no-default-features

# Criterion benchmarks
bench:
	cd flowstate-core && cargo bench --no-default-features

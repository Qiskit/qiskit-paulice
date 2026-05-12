.PHONY: clean
clean:
	rm -rf .mypy_cache .ruff_cache build Cargo.lock htmlcov .coverage docs/*/.ipynb_checkpoints python/*/__pycache__ python/*/*/__pycache__ target */__pycache__ python/qiskit_paulice/_internal/*.so

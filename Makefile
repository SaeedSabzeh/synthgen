.PHONY: install test lint fmt typecheck demo clean

install:
	pip install -e ".[pandas,dev]"

test:
	pytest --cov=synthgen --cov-report=term-missing

lint:
	ruff check .

fmt:
	ruff check --fix . && ruff format .

typecheck:
	mypy src/synthgen

demo:
	synthgen --schema person --rows 30 --out out/people.csv --report -v

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build out
	find . -name __pycache__ -type d -exec rm -rf {} +

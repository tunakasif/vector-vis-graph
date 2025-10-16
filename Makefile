PROJECT_NAME:=vector_vis_graph
EXECUTER:=uv run

all: format lint security test

install:
	git init
	uv sync --all-groups
	$(EXECUTER) pre-commit install

clean:
	rm -rf .hypothesis .ipynb_checkpoints .mypy_cache .pytest_cache .coverage htmlcov
	$(EXECUTER) ruff clean

test:
	 $(EXECUTER) pytest -n auto --cov-report term-missing --cov-report html --cov $(PROJECT_NAME)/

format:
	$(EXECUTER) ruff format .

lint:
	$(EXECUTER) ruff check . --fix
	$(EXECUTER) mypy .

security:
	$(EXECUTER) bandit -r $(PROJECT_NAME)/


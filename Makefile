PROJECT_NAME:=vector_vis_graph
EXECUTER:=poetry run

all: format lint security test requirements

install:
	git init
	$(EXECUTER) poetry install
	$(EXECUTER) pre-commit install

clean:
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov
	$(EXECUTER) ruff clean

requirements:
	poetry export -f requirements.txt -o requirements.txt --with dev --without-hashes

test:
	$(EXECUTER) pytest --cov-report term-missing --cov-report html --cov $(PROJECT_NAME)/

format:
	$(EXECUTER) ruff format .

lint:
	$(EXECUTER) ruff check . --fix
	$(EXECUTER) mypy .

security:
	$(EXECUTER) bandit -r $(PROJECT_NAME)/


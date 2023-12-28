PROJECT_NAME:=vector_vis_graph
EXECUTER:=poetry run
WORKER_COUNT:=$(shell nproc)

all: format lint security test requirements

install:
	git init
	$(EXECUTER) poetry install
	$(EXECUTER) pre-commit install

clean:
	rm -rf .hypothesis .ipynb_checkpoints .mypy_cache .pytest_cache .coverage htmlcov
	$(EXECUTER) ruff clean

requirements:
	poetry export -f requirements.txt -o requirements.txt --with dev,test --without-hashes

test:
	$(EXECUTER) pytest -n $(WORKER_COUNT) --cov-report term-missing --cov-report html --cov $(PROJECT_NAME)/

format:
	$(EXECUTER) ruff format .

lint:
	$(EXECUTER) ruff check . --fix
	$(EXECUTER) mypy .

security:
	$(EXECUTER) bandit -r $(PROJECT_NAME)/


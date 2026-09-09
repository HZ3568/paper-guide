.PHONY: help install dev run test integration-tests eval intent-eval clarify-eval route-eval lint format

ENV_NAME = paper-guide

help:
	@echo 'Targets:'
	@echo '  install             Create or update the conda environment (python 3.13 + deps)'
	@echo '  dev                 Alias for install (full dev environment)'
	@echo '  run                 Start the FastAPI web app (uvicorn)'
	@echo '  test                Run unit tests'
	@echo '  integration-tests   Run integration tests'
	@echo '  eval                Run RAG retrieval eval (baseline vs pipeline)'
	@echo '  intent-eval         Run Search intent golden set (live model)'
	@echo '  clarify-eval        Run clarification judgment golden set (live model)'
	@echo '  route-eval          Run Supervisor routing golden set (live model)'
	@echo '  lint                Run Ruff checks'
	@echo '  format              Format with Ruff'

install:
	conda env update -f environment.yml --prune || conda env create -f environment.yml

dev: install

run:
	conda run --no-capture-output -n $(ENV_NAME) uvicorn api:app --reload --host 0.0.0.0 --port 8000

test:
	conda run -n $(ENV_NAME) python -m pytest tests/unit_tests -q

integration-tests:
	conda run -n $(ENV_NAME) python -m pytest tests/integration_tests -q

eval:
	conda run -n $(ENV_NAME) python -m paper_guide.eval.rag_eval

intent-eval:
	conda run -n $(ENV_NAME) python -m paper_guide.eval.intent_eval

clarify-eval:
	conda run -n $(ENV_NAME) python -m paper_guide.eval.clarify_eval

route-eval:
	conda run -n $(ENV_NAME) python -m paper_guide.eval.route_eval

lint:
	conda run -n $(ENV_NAME) python -m ruff check src tests api.py

format:
	conda run -n $(ENV_NAME) python -m ruff format src tests api.py

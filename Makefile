PYTHON ?= python

.PHONY: help setup data train evaluate predict app test sanity pipeline

help:
	@echo "Available targets:"
	@echo "  setup    - create virtual environment and install requirements (manual activation required)"
	@echo "  data     - build processed train/test datasets"
	@echo "  train    - train and benchmark models"
	@echo "  evaluate - generate stakeholder/explainability artifacts"
	@echo "  predict  - generate submission file"
	@echo "  app      - run Streamlit app"
	@echo "  sanity   - run pre-deploy sanity checks"
	@echo "  test     - run pytest suite"
	@echo "  pipeline - run data + train + evaluate"

setup:
	$(PYTHON) -m venv .venv
	@echo "Activate venv then run: pip install -r requirements.txt"

data:
	$(PYTHON) -m src.make_dataset

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

predict:
	$(PYTHON) -m src.predict --out submission.csv

app:
	streamlit run streamlit_app.py

sanity:
	$(PYTHON) -m src.sanity_check

test:
	pytest -q

pipeline: data train evaluate

PYTHON ?= python3
APP_ENV ?= .venv
SFT_ENV ?= .venv-sft

.PHONY: setup-app setup-sft run check export-sft split-sft prepare-sft

setup-app:
	$(PYTHON) -m venv $(APP_ENV)
	. $(APP_ENV)/bin/activate && pip install --upgrade pip && pip install -e .

setup-sft:
	$(PYTHON) -m venv $(SFT_ENV)
	. $(SFT_ENV)/bin/activate && pip install --upgrade pip && pip install "llamafactory>=0.9.0"

run:
	. $(APP_ENV)/bin/activate && uvicorn agent_rag.main:app --reload

check:
	. $(APP_ENV)/bin/activate && python -m compileall src && python -m py_compile scripts/*.py

export-sft:
	. $(APP_ENV)/bin/activate && python scripts/export_sft_dataset.py \
		--output workspace_data/sft/qwen2_5_vl_all.jsonl \
		--accepted-only \
		--exclude-layout-questions

split-sft:
	. $(APP_ENV)/bin/activate && python scripts/split_sft_dataset.py \
		--input workspace_data/sft/qwen2_5_vl_all.jsonl \
		--train-output workspace_data/sft/qwen2_5_vl_train.jsonl \
		--eval-output workspace_data/sft/qwen2_5_vl_eval.jsonl \
		--eval-ratio 0.1

prepare-sft:
	. $(APP_ENV)/bin/activate && python scripts/launch_sft.py \
		--config configs/sft/qwen2_5_vl_pdfqa.yaml \
		--train-file workspace_data/sft/qwen2_5_vl_train.jsonl \
		--run-name qwen2_5_vl_first_run


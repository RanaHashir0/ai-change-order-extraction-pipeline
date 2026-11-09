# AI-Powered Construction Change Order Extraction Pipeline
# Compatible with GNU Make and Windows mingw32-make

PYTHON ?= python

ifeq ($(OS),Windows_NT)
    SET_PYTHONPATH = set PYTHONPATH=src&&
else
    SET_PYTHONPATH = PYTHONPATH=src
endif

.PHONY: all help extract test schema clean

help:
	@echo AI-Powered Construction Change Order Extraction Pipeline
	@echo --------------------------------------------------------
	@echo Usage:
	@echo   make extract FILE=path/to/file.pdf   Extract structured JSON from document
	@echo   make test                             Run complete test suite
	@echo   make schema                           Regenerate JSON Schema Draft 2020-12
	@echo   make clean                            Remove cache and temporary files

extract:
ifndef FILE
	$(error FILE parameter is required. Usage: make extract FILE=<path_to_file>)
endif
	@$(SET_PYTHONPATH) $(PYTHON) -m change_order_extractor.cli "$(FILE)"

test:
	@$(SET_PYTHONPATH) $(PYTHON) -m pytest

schema:
	@$(SET_PYTHONPATH) $(PYTHON) -c "from change_order_extractor.models import export_json_schema; export_json_schema()"

clean:
	@$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if p.exists()]"
	@$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache') if p.exists()]"

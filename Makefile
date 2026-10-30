# Makefile - install, reproduce, test. Nothing here needs a GPU or the datasets.
#
#   make setup && make data REMOTE=<rclone remote> && make reproduce && make test

VENV ?= .venv
PY    = $(CURDIR)/$(VENV)/bin/python
CODE  = src

# Auslan-Daily is not redistributed. `make data` copies its translation spreadsheet from the dataset's official,
# public Google Drive folder with rclone. REMOTE is the name of any Google Drive remote made with `rclone config`.
REMOTE ?= gdrive
RCLONE ?= rclone
AUSLAN_DAILY_FOLDER = 17E5wgq1ig7-WynNskG-8tTMoOnGlVh4V
SHEET = Dataset Split Table/Sign Language Translation/Ausan-Daily Communication/Auslan-Daily_Communication.xlsx
DATA_DIR = data/auslan-daily/translation

.PHONY: help setup data references headline reproduce test smoke clean venvcheck

help:
	@echo "make setup      - create $(VENV) and install requirements.txt"
	@echo "make data       - fetch the Auslan-Daily spreadsheet with rclone (REMOTE=<remote>) and restore the reference sentences"
	@echo "make headline   - the reported system's scores and the decision rule (seconds)"
	@echo "make reproduce  - re-derive the reported numbers from the saved outputs (every runnable check in REPRODUCING.md)"
	@echo "make test       - run the tests in tests/"
	@echo "make smoke      - train and decode on synthetic clips (needs no data)"
	@echo "make clean      - remove $(VENV) and caches"

# Fail with an instruction rather than a bare "No such file or directory".
venvcheck:
	@test -x "$(PY)" || { echo "error: $(VENV) not found. Run 'make setup' first."; exit 1; }

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

# Fetch the spreadsheet from the dataset's official folder, then restore the reference sentences (see ATTRIBUTION.md).
data: venvcheck
	$(RCLONE) copy --drive-root-folder-id $(AUSLAN_DAILY_FOLDER) "$(REMOTE):$(SHEET)" $(DATA_DIR)
	$(MAKE) references

# Put the reference sentences back into results/ from the spreadsheet, verify them and write the text manifests.
references: venvcheck
	$(PY) $(CODE)/tools/restore_references.py

# The reported system's scores, the decision-rule self-test and the confirmatory family (a few seconds).
headline: venvcheck
	$(PY) $(CODE)/tools/run_claims.py REPRODUCING.md --only A1,A10,A11

# Every check in REPRODUCING.md (A1 to A47; the bootstraps dominate the run time). Run `make data` first.
# It passes when every check that exists reproduces its recorded output; a claim that has no check does not make it fail.
reproduce: venvcheck
	$(PY) $(CODE)/tools/run_claims.py REPRODUCING.md

test: venvcheck
	$(PY) -m pytest tests -q

smoke: venvcheck
	cd $(CODE) && $(PY) smoke_test.py

clean:
	rm -rf $(VENV) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

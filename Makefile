PY ?= python

install:
	pip install -r requirements.txt

data:
	$(PY) scripts/download_data.py

test:
	pytest -q tests

experiments:
	bash scripts/reproduce.sh

analysis:
	$(PY) scripts/analyze.py oulad
	$(PY) scripts/analyze.py act-mooc
	$(PY) scripts/dataset_stats.py
	$(PY) scripts/val_variant.py
	$(PY) scripts/make_results.py paper
	$(PY) scripts/make_tables.py paper/tables
	$(PY) scripts/make_numbers.py paper/numbers.tex
	$(PY) scripts/update_readme.py README.md

benchmark:
	$(PY) scripts/benchmark_efficiency.py

# Regenerate every number and table of the article from results/ and compare with the committed copies in paper/
verify:
	@T=$$(mktemp -d) && $(PY) scripts/make_numbers.py $$T/numbers.tex > /dev/null && diff $$T/numbers.tex paper/numbers.tex \
	  && OPAL_FIG_DIR=$$T/figures $(PY) scripts/make_results.py $$T/paper > /dev/null && $(PY) scripts/make_tables.py $$T/paper/tables > /dev/null \
	  && diff -r $$T/paper/tables paper/tables && rm -rf $$T && echo "OK: all numbers and tables of the article regenerate identically"

figures:
	$(PY) scripts/fig_protocol.py
	$(PY) scripts/fig_architecture.py

.PHONY: install data test experiments analysis figures benchmark verify

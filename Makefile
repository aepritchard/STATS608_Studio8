PYTHON ?= python3
REPS ?= 100
SEED ?= 20261015

.PHONY: all simulate analyze figures test clean

all:
	$(MAKE) simulate
	$(MAKE) analyze
	$(MAKE) figures

simulate:
	$(PYTHON) simulate.py --reps $(REPS) --seed $(SEED)

figures:
	$(PYTHON) analyze_results.py

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf results .pytest_cache

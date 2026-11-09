PYTHON ?= python3
INDEX  ?= indexes/wiki
DATA   ?= data
LIMIT  ?= 100000
PORT   ?= 8000
WIKI_DUMP := $(DATA)/raw/simplewiki-latest-pages-articles.xml.bz2
WIKI_JSONL := $(DATA)/simplewiki.jsonl
CRAWL_SEEDS ?= https://quotes.toscrape.com/ https://books.toscrape.com/

.PHONY: help install test lint format data crawl index index-crawl serve benchmark evaluate docker clean

help:  ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install the package with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## Run the test suite
	$(PYTHON) -m pytest

lint:  ## ruff + black --check
	ruff check src tests benchmarks
	black --check src tests benchmarks

format:  ## Auto-format
	ruff check --fix src tests benchmarks
	black src tests benchmarks

$(WIKI_DUMP):
	mkdir -p $(DATA)/raw
	curl -fL --retry 5 -C - -o $@ https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2

$(WIKI_JSONL): $(WIKI_DUMP)
	$(PYTHON) -m minisearch wiki --dump $< --out $@

data: $(WIKI_JSONL)  ## Download + convert Simple English Wikipedia

crawl:  ## Politely crawl the demo seed sites into data/crawl.jsonl
	$(PYTHON) -m minisearch crawl --seeds $(CRAWL_SEEDS) --out $(DATA)/crawl.jsonl --max-pages 300 --delay 1.0

index: $(WIKI_JSONL)  ## Index LIMIT Wikipedia articles into INDEX
	$(PYTHON) -m minisearch index --index $(INDEX) --input $(WIKI_JSONL) --limit $(LIMIT) --force-merge

index-crawl:  ## Index the crawled pages
	$(PYTHON) -m minisearch index --index indexes/crawl --input $(DATA)/crawl.jsonl --force-merge

serve:  ## Run API + UI on PORT
	$(PYTHON) -m minisearch serve --index $(INDEX) --host 0.0.0.0 --port $(PORT)

benchmark: $(WIKI_JSONL)  ## Throughput, size, latency, memory benchmarks + charts
	$(PYTHON) benchmarks/run_benchmarks.py --input $(WIKI_JSONL) --limit $(LIMIT)

evaluate:  ## Relevance evaluation (Cranfield, SciFact): P@10, R@10, MAP, nDCG@10
	$(PYTHON) benchmarks/evaluate.py

docker:  ## Build the Docker image
	docker build -t minisearch .

clean:  ## Remove caches (keeps data and indexes)
	rm -rf .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

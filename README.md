# RAG Search Engine: Keyword Retrieval CLI

A lightweight, high-performance local keyword search engine written in Python. It indexes textual documents and supports document retrieval using both Boolean Term Frequency / TF-IDF scoring and the industry-standard **Okapi BM25** ranking algorithm.

Designed to serve as the sparse retrieval component of a **RAG (Retrieval-Augmented Generation)** pipeline, this engine operates offline with no external service dependencies.

---

## Features

- **Okapi BM25 Ranking**: State-of-the-art sparse keyword ranking based on term saturation and document length normalization.
- **TF-IDF Diagnostics**: Commands to extract raw TF, IDF, and TF-IDF statistics for search terms across documents.
- **Stemming & Text Normalization**: Automated case folding, punctuation stripping, and token stemming using the NLTK `PorterStemmer` pipeline.
- **Stopwords Filtering**: Support for customizable stopword lists to improve retrieval quality by filtering out highly frequent but non-informative words.
- **Fast Serialization**: Index data structures are fully serialized using `pickle` for instant loading during search operations.

---

## Installation

This project is built using Python (version `>=3.13`) and requires [nltk](https://www.nltk.org/).

### Using `uv` (Recommended)

If you have `uv` installed, you can run commands directly without manual environment setup:

```bash
# Install dependencies and run commands
uv run cli/keyword_search_cli.py --help
```

### Using standard virtual environment

Alternatively, create a virtual environment and install the dependencies:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r pyproject.toml
```

---

## Quickstart

Start by building the search index from the movie database, then perform your first search query:

```bash
# 1. Build the index from the default movies dataset
uv run cli/keyword_search_cli.py build

# 2. Perform a BM25 keyword search
uv run cli/keyword_search_cli.py bm25search "blue avatar alien" --limit 3
```

---

## CLI Command Reference

### Index Operations

#### `build`
Builds and serializes the inverted index and term frequency map from a JSON dataset.
```bash
uv run cli/keyword_search_cli.py build [data_file] [index_dir]
```
- **`data_file`** *(Optional, default: `data/movies.json`)*: Path to the movie dataset JSON.
- **`index_dir`** *(Optional, default: `cache`)*: Directory to save serialized `.pkl` files (`index.pkl`, `docmap.pkl`, `term_frequencies.pkl`).

---

### Search Operations

#### `bm25search` (Recommended)
Searches the collection using Okapi BM25 scoring.
```bash
uv run cli/keyword_search_cli.py bm25search "your search query" [--limit LIMIT] [index_dir]
```
- **`--limit`** *(Optional, default: `5`)*: Maximum number of search results to return.
- **`index_dir`** *(Optional, default: `cache`)*: Directory to load index files from.

#### `search`
Performs a standard Boolean / TF-IDF search, filtering out common stopwords.
```bash
uv run cli/keyword_search_cli.py search "your search query" [index_dir] [num_results]
```

---

### Diagnostic Operations

These commands allow you to inspect the index internals for debugging or verification.

#### `tf`
Retrieve the raw frequency of a stemmed term in a specific document.
```bash
uv run cli/keyword_search_cli.py tf <doc_id> <term> [index_dir]
```

#### `idf`
Retrieve the standard Inverse Document Frequency (IDF) of a stemmed term.
```bash
uv run cli/keyword_search_cli.py idf <term> [index_dir]
```

#### `tfidf`
Retrieve the combined TF-IDF score for a stemmed term in a document.
```bash
uv run cli/keyword_search_cli.py tfidf <doc_id> <term> [index_dir]
```

#### `bm25tf`
Retrieve the length-normalized Okapi BM25 term frequency score.
```bash
uv run cli/keyword_search_cli.py bm25tf <doc_id> <term> [k1] [index_dir]
```

#### `bm25idf`
Retrieve the Okapi BM25 Inverse Document Frequency (IDF) score.
```bash
uv run cli/keyword_search_cli.py bm25idf <term> [index_dir]
```

---

## Project Structure

```
.
├── cli/
│   ├── __init__.py
│   ├── constants.py           # BM25 parameters: k1=1.5, b=0.75
│   └── keyword_search_cli.py  # Core search engine class & CLI entrypoint
├── data/
│   ├── movies.json            # Default dataset (~26MB, movies catalog)
│   └── stopwords.txt          # Default stopwords list
├── docs/
│   └── api.md                 # Detailed developer API documentation
├── tests/
│   └── test_keyword_search.py # Comprehensive unit test suite
├── pyproject.toml             # Project dependency specification
└── README.md                  # This file
```

---

## Running Unit Tests

The project includes a comprehensive unit test suite covering text preprocessing, indexing correctness, TF-IDF & BM25 score calculations, index persistence, and CLI command execution.

Run the tests using standard Python `unittest`:

```bash
uv run python -m unittest discover -s tests
```
or
```bash
python3 -m unittest discover -s tests
```

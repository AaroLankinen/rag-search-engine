# Developer API Reference - Keyword Search Engine

This document provides a technical guide for developers who want to use or extend the `rag-search-engine` CLI and library components.

---

## Architecture Overview

The project is structured around a central class `InvertedIndex` which performs two primary tasks:
1. **Indexing**: Preprocessing raw textual documents and organizing them into an inverted mapping (terms to document IDs) along with document statistics.
2. **Retrieval**: Query parsing and document scoring using either standard Boolean Search, TF-IDF representations, or the Okapi BM25 ranking algorithm.

### Text Preprocessing Pipeline
All text processing (for both documents and search queries) undergoes the following transformations inside [preprocess_text](file:///home/aarol/workspace/rag-search-engine/cli/keyword_search_cli.py#L147-L150):
1. **Case Normalization**: Converts all text to lowercase.
2. **Punctuation Removal**: Strips all punctuation marks.
3. **Stemming**: Applies NLTK's `PorterStemmer` to stem words to their base form (e.g., "running", "runs" -> "run").

---

## API Reference

### `InvertedIndex`

The [InvertedIndex](file:///home/aarol/workspace/rag-search-engine/cli/keyword_search_cli.py#L22) class coordinates index storage, term weighting, and search operations.

```python
class InvertedIndex:
    index: dict[str, list[str]]
    doc_map: dict[str, str]
    term_frequencies: dict[str, collections.Counter]
```

#### `__init__(self, documents: list[str] | dict[str, str])`
Instantiates a new InvertedIndex.
- **`documents`**: Either a list of document texts (in which case document IDs are auto-assigned as `"0"`, `"1"`, etc.), or a dictionary mapping unique string document IDs to their contents.

#### `get_tf(self, doc_id: str, term: str) -> int`
Retrieves the raw term frequency (number of occurrences) of a preprocessed term in the specified document.
- **`doc_id`**: The document ID to search.
- **`term`**: The preprocessed target term.

#### `get_bm25_idf(self, term: str) -> float`
Calculates the Okapi BM25 Inverse Document Frequency (IDF) score for a term:
$$\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$$
where $N$ is the total document count and $n(q_i)$ is the number of documents containing the term.

#### `avg_doc_len` (property) -> `float`
Returns the average document length (in words) across the indexed collection. Results are cached after the first compute.

#### `get_bm25_tf(self, doc_id: str, term: str, k1: float = K1, b: float = B) -> float`
Calculates the scaled BM25 term frequency term:
$$\text{BM25\_TF}(d, t) = \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$
- **`k1`**: Controls term frequency scaling (defaults to `1.5` from `cli/constants.py`).
- **`b`**: Controls document length normalization scaling (defaults to `0.75` from `cli/constants.py`).

#### `bm25(self, doc_id: str, term: str, k1: float = K1, b: float = B) -> float`
Computes the cumulative Okapi BM25 score for a multi-token query term against a document.

#### `bm25_search(self, query: str, limit: int = 10) -> list[str]`
Searches the collection using Okapi BM25 scoring.
- **Returns**: A list of up to `limit` document IDs sorted by BM25 score in descending order. Ties are resolved by sorting document IDs in ascending integer order.

#### `search(self, query: str) -> list[str]`
Performs a standard Boolean Union search.
- **Returns**: A list of document IDs that match *any* of the tokens in the query. Note that duplicates are not removed by this raw method.

#### `save(self, directory: str) -> None`
Serializes and saves `index`, `doc_map`, and `term_frequencies` state to `.pkl` files in the target directory using `pickle`.

#### `load(self, directory: str) -> None`
Deserializes index files (`index.pkl`, `docmap.pkl`, `term_frequencies.pkl`) from the target directory and populates the instance variables.

---

### Utility Functions

#### `preprocess_text(text: str) -> list[str]`
Splits text into lowercase, stripped, and stemmed tokens using PorterStemmer.

#### `tokenize_term(term: str) -> str`
Tokenizes a single term, returning the first stemmed token. Useful for CLI arguments query normalization.

#### `load_stopwords(filepath: str) -> set[str]`
Reads a file containing stopwords (one per line), tokenizes/stems each, and returns them as a set of stemmed tokens. If the file is not found, prints a warning and returns an empty set.

---

## Library Usage Example

You can use the index programmatic as follows:

```python
from cli.keyword_search_cli import InvertedIndex, preprocess_text

# 1. Initialize with documents
documents = {
    "1": "The movie was filled with action and suspense.",
    "2": "A romantic drama set in the 19th century.",
    "3": "Action-packed thriller with science fiction themes."
}

index = InvertedIndex(documents)

# 2. Perform a BM25 search
query = "action thriller"
results = index.bm25_search(query, limit=2)
print("BM25 Results:", results)  # Expected: ['3', '1']

# 3. Retrieve TF and BM25 statistics
stemmed_term = "action"
tf = index.get_tf("3", stemmed_term)
score = index.bm25("3", stemmed_term)

print(f"Term '{stemmed_term}' in Doc 3 -> TF: {tf}, BM25 Score: {score:.4f}")

# 4. Save index for subsequent use
index.save("./cache")
```

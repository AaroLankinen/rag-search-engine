#!/usr/bin/env python3
import argparse
import json
import sys
import string
import collections
import math
from nltk.stem import PorterStemmer

try:
    from cli.constants import K1, B
except ImportError:
    from constants import K1, B

stemmer = PorterStemmer()

import pickle

# --- Inverted Index and Document Map ---
# @attribute index: maps a term to a list of document IDs
# @attribute doc_map: maps a document ID to its content
class InvertedIndex:
    index: dict[str, list[str]]
    doc_map: dict[str, str]
    term_frequencies: dict[str, collections.Counter]

    def __init__(self, documents: list[str] | dict[str, str]):
        self.index = {}
        self.doc_map = {}
        self.term_frequencies = collections.defaultdict(collections.Counter)
        self._avg_doc_len = None
        if isinstance(documents, dict):
            for doc_id, doc in documents.items():
                self.__add_document(doc_id, doc)
        else:
            for doc in documents:
                self.__add_document(str(len(self.doc_map)), doc)
    
    # @function get_tf: get the term frequency for a term in a document
    # @return: int
    def get_tf(self, doc_id: str, term: str) -> int:
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id].get(term, 0)

    # @function get_bm25_idf: get the Okapi BM25 IDF value for a term
    # @return: float
    def get_bm25_idf(self, term: str) -> float:
        total_documents = len(self.doc_map)
        term_match_doc_count = len(self.index.get(term, []))
        return math.log((total_documents - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)

    @property
    def avg_doc_len(self) -> float:
        if self._avg_doc_len is None:
            if not self.doc_map:
                return 0.0
            self._avg_doc_len = sum(len(preprocess_text(doc)) for doc in self.doc_map.values()) / len(self.doc_map)
        return self._avg_doc_len

    # @function get_bm25_tf: get the Okapi BM25 TF value for a term
    # @return: float
    def get_bm25_tf(self, doc_id: str, term: str, k1: float = K1, b: float = B) -> float:
        doc_len = len(preprocess_text(self.doc_map[doc_id]))
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / self.avg_doc_len))    

    # @function bm25: get the Okapi BM25 score for a term in a document
    # @return: float
    def bm25(self, doc_id: str, term: str, k1: float = K1, b: float = B) -> float:
        score = 0
        for token in preprocess_text(term):
            score += self.get_bm25_tf(doc_id, token, k1, b) * self.get_bm25_idf(token)
        return score    

    # @function bm25_search: search for documents containing the query
    # @return: list of document IDs 
    def bm25_search(self, query: str, limit: int = 10) -> list[str]:
        query_tokens = preprocess_text(query)
        results = {}
        for token in query_tokens:
            for doc_id in self.get_documents(token):
                results[doc_id] = self.bm25(doc_id, query)
        sorted_results = sorted(results.items(), key=lambda x: (x[1], -int(x[0])), reverse=True)
        return [doc_id for doc_id, score in sorted_results[:limit]] 

    # @function __add_document: add a document to the inverted index
    # @return: None 
    def __add_document(self, doc_id: str, doc: str) -> None:
        self.doc_map[doc_id] = doc
        for token in preprocess_text(doc):
            if token not in self.index:
                self.index[token] = []
            if doc_id not in self.index[token]:
                self.index[token].append(doc_id)
            self.term_frequencies[doc_id][token] += 1

    # @function get_documents: get documents for a term
    # @return: list of document IDs
    def get_documents(self, term: str) -> list[str]:
        terms = preprocess_text(term)
        if not terms:
            return []
        return self.index.get(terms[0], [])

    # @function search: search for documents containing the query
    # @return: list of document IDs 
    def search(self, query: str) -> list[str]:
        query_tokens = preprocess_text(query)
        results = []
        for token in query_tokens:
            results.extend(self.get_documents(token))
        return results
    
    # @function save: save the inverted index and document map to files
    # @return: None 
    def save(self, directory: str) -> None:
        import os
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        with open(os.path.join(directory, "docmap.pkl"), "wb") as f:
            pickle.dump(self.doc_map, f)
        with open(os.path.join(directory, "term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)

    # @function load: load the inverted index and document map from files
    # @return: None
    def load(self, directory: str) -> None:
        import os
        try:
            with open(os.path.join(directory, "index.pkl"), "rb") as f:
                self.index = pickle.load(f)
            with open(os.path.join(directory, "docmap.pkl"), "rb") as f:
                self.doc_map = pickle.load(f)
            with open(os.path.join(directory, "term_frequencies.pkl"), "rb") as f:
                self.term_frequencies = pickle.load(f)
            self._avg_doc_len = None
        except FileNotFoundError:
            print(f"Error: Index files not found in '{directory}'.", file=sys.stderr)
            sys.exit(1)


# --- Helper Functions ---
# @function preprocess_text: preprocess the input text
# @return: list of tokens
def preprocess_text(text: str) -> list[str]:
    translator = str.maketrans("", "", string.punctuation)
    clean_text = text.lower().translate(translator)
    return [stemmer.stem(tok) for tok in clean_text.split()]


# @function tokenize_term: tokenize a single search term
# @return: str
def tokenize_term(term: str) -> str:
    tokens = preprocess_text(term)
    return tokens[0] if tokens else ""


# @function load_stopwords: load stopwords from a file
# @return: set of stopwords
def load_stopwords(filepath: str) -> set[str]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            words = f.read().splitlines()
    except FileNotFoundError:
        print(f"Warning: Data file '{filepath}' not found.", file=sys.stderr)
        return set()
    
    stop_words = set()
    for word in words:
        for token in preprocess_text(word):
            stop_words.add(token)
    return stop_words


# --- Commands ---
# @function build_command: build and save the inverted index
def build_command(data_file: str, index_dir: str) -> None:
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            movies_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Data file '{data_file}' is not valid JSON.", file=sys.stderr)
        sys.exit(1)

    if isinstance(movies_data, dict):
        movies = movies_data.get("movies", [])
    elif isinstance(movies_data, list):
        movies = movies_data
    else:
        movies = []

    documents = {
        str(movie["id"]): f"{movie.get('title', '')}\n{movie.get('description', '')}"
        for movie in movies
        if "id" in movie
    }
    inverted_index = InvertedIndex(documents)
    inverted_index.save(index_dir)




def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("index_dir", type=str, nargs="?", default="cache", help="Directory containing index files")
    search_parser.add_argument("num_results", type=int, nargs="?", default=5, help="Number of results to return")

    build_parser = subparsers.add_parser("build", help="Build and save the inverted index")
    build_parser.add_argument("data_file", type=str, nargs="?", default="data/movies.json", help="Path to data file")
    build_parser.add_argument("index_dir", type=str, nargs="?", default="cache", help="Directory to save index files")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency of a term in a document")
    tf_parser.add_argument("doc_id", type=str, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")
    tf_parser.add_argument("index_dir", type=str, nargs="?", default="cache", help="Directory containing index files")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency of a term")
    idf_parser.add_argument("term", type=str, help="Search term")
    idf_parser.add_argument("index_dir", type=str, nargs="?", default="cache", help="Directory containing index files")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF value of a term in a document")
    tfidf_parser.add_argument("doc_id", type=str, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search term")
    tfidf_parser.add_argument("index_dir", type=str, nargs="?", default="cache", help="Directory containing index files")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get Okapi BM25 IDF value of a term")
    bm25idf_parser.add_argument("term", type=str, help="Search term")
    bm25idf_parser.add_argument("index_dir", type=str, nargs="?", default="cache", help="Directory containing index files")

    bm25tf_parser = subparsers.add_parser("bm25tf", help="Get Okapi BM25 TF value of a term in a document")
    bm25tf_parser.add_argument("doc_id", type=str, help="Document ID")
    bm25tf_parser.add_argument("term", type=str, help="Search term")
    bm25tf_parser.add_argument("k1", type=float, nargs="?", default=K1, help="BM25 k1 parameter")
    bm25tf_parser.add_argument("index_dir", type=str, nargs="?", default="cache", help="Directory containing index files")

    bm25search_parser = subparsers.add_parser("bm25search", help="Perform Okapi BM25 search")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
    bm25search_parser.add_argument("index_dir", type=str, nargs="?", default="cache", help="Directory containing index files")

    args = parser.parse_args()
    stop_words = load_stopwords("data/stopwords.txt")

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            inverted_index = InvertedIndex([])
            inverted_index.load(args.index_dir)

            results = {}
            query_tokens = [tok for tok in preprocess_text(args.query) if tok not in stop_words]
            for q_tok in query_tokens:
                matched_docs = set()
                for term in inverted_index.index:
                    if q_tok in term:
                        matched_docs.update(inverted_index.index[term])
                for doc_id in matched_docs:
                    results[doc_id] = results.get(doc_id, 0) + 1

            # Sort by score in descending order, and then by doc_id in ascending order for ties
            sorted_results = sorted(results.items(), key=lambda x: (x[1], -int(x[0])), reverse=True)

            for doc_id, score in sorted_results[:args.num_results]:
                full_doc = inverted_index.doc_map.get(doc_id, "Unknown Title")
                title = full_doc.split("\n", 1)[0]
                print(f"Document ID: {doc_id}, Title: {title}")
        case "build":
            build_command(args.data_file, args.index_dir)
        case "tf":
            inverted_index = InvertedIndex([])
            inverted_index.load(args.index_dir)
            token = tokenize_term(args.term)
            tf = inverted_index.get_tf(args.doc_id, token)
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf}")
        case "idf":
            inverted_index = InvertedIndex([])
            inverted_index.load(args.index_dir)
            token = tokenize_term(args.term)
            total_doc_count = len(inverted_index.doc_map)
            term_match_docs = inverted_index.index.get(token, [])
            term_match_doc_count = len(term_match_docs)
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            inverted_index = InvertedIndex([])
            inverted_index.load(args.index_dir)
            token = tokenize_term(args.term)
            tf = inverted_index.get_tf(args.doc_id, token)
            total_doc_count = len(inverted_index.doc_map)
            term_match_docs = inverted_index.index.get(token, [])
            term_match_doc_count = len(term_match_docs)
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            tfidf = tf * idf
            print(f"TF-IDF of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}")
        case "bm25idf":
            inverted_index = InvertedIndex([])
            inverted_index.load(args.index_dir)
            token = tokenize_term(args.term)
            bm25idf = inverted_index.get_bm25_idf(token)
            print(f"BM25 IDF of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            inverted_index = InvertedIndex([])
            inverted_index.load(args.index_dir)
            token = tokenize_term(args.term)
            bm25tf = inverted_index.get_bm25_tf(args.doc_id, token, k1=args.k1)
            print(f"BM25 TF of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            inverted_index = InvertedIndex([])
            inverted_index.load(args.index_dir)
            doc_ids = inverted_index.bm25_search(args.query, limit=args.limit)
            for i, doc_id in enumerate(doc_ids, start=1):
                full_doc = inverted_index.doc_map.get(doc_id, "Unknown Title")
                title = full_doc.split("\n", 1)[0]
                score = inverted_index.bm25(doc_id, args.query)
                print(f"{i}. ({doc_id}) {title} - Score: {score:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
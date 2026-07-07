import argparse
import json
import sys
import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

import pickle

class InvertedIndex:
    index: dict[str, list[str]]
    doc_map: dict[str, str]

    def __init__(self, documents: list[str] | dict[str, str]):
        self.index = {}
        self.doc_map = {}
        if isinstance(documents, dict):
            for doc_id, doc in documents.items():
                self.__add_document(doc_id, doc)
        else:
            for doc in documents:
                self.__add_document(str(len(self.doc_map)), doc)
    
    def __add_document(self, doc_id: str, doc: str) -> None:
        self.doc_map[doc_id] = doc
        for token in preprocess_text(doc):
            if token not in self.index:
                self.index[token] = []
            if doc_id not in self.index[token]:
                self.index[token].append(doc_id)
    
    def get_documents(self, term: str) -> list[str]:
        terms = preprocess_text(term)
        if not terms:
            return []
        return self.index.get(terms[0], [])

    def search(self, query: str) -> list[str]:
        query_tokens = preprocess_text(query)
        results = []
        for token in query_tokens:
            results.extend(self.get_documents(token))
        return results

    def save(self, directory: str) -> None:
        import os
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        with open(os.path.join(directory, "docmap.pkl"), "wb") as f:
            pickle.dump(self.doc_map, f)

def preprocess_text(text: str) -> list[str]:
    translator = str.maketrans("", "", string.punctuation)
    clean_text = text.lower().translate(translator)
    return [stemmer.stem(tok) for tok in clean_text.split()]

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

    documents = {str(movie["id"]): movie.get("description", "") for movie in movies if "id" in movie}
    inverted_index = InvertedIndex(documents)
    inverted_index.save(index_dir)
    
    docs = inverted_index.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("data_file", type=str, nargs="?", default="data/movies.json", help="Path to data file")
    search_parser.add_argument("num_results", type=int, nargs="?", default=5, help="Number of results to return")

    build_parser = subparsers.add_parser("build", help="Build and save the inverted index")
    build_parser.add_argument("data_file", type=str, nargs="?", default="data/movies.json", help="Path to data file")
    build_parser.add_argument("index_dir", type=str, nargs="?", default="cache", help="Directory to save index files")

    args = parser.parse_args()
    stop_words = load_stopwords("data/stopwords.txt")

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            try:
                with open(args.data_file, "r", encoding="utf-8") as f:
                    movies_data = json.load(f)
            except FileNotFoundError:
                print(f"Error: Data file '{args.data_file}' not found.", file=sys.stderr)
                sys.exit(1)
            except json.JSONDecodeError:
                print(f"Error: Data file '{args.data_file}' is not valid JSON.", file=sys.stderr)
                sys.exit(1)

            if isinstance(movies_data, dict):
                movies = movies_data.get("movies", [])
            elif isinstance(movies_data, list):
                movies = movies_data
            else:
                movies = []

            results = []
            query_tokens = [tok for tok in preprocess_text(args.query) if tok not in stop_words]
            for movie in movies:
                if not isinstance(movie, dict):
                    continue
                title = movie.get("title", "")
                title_tokens = [tok for tok in preprocess_text(title) if tok not in stop_words]
                
                # Count how many query tokens appear in any title token
                score = sum(1 for q_tok in query_tokens if any(q_tok in t_tok for t_tok in title_tokens))
                if score > 0:
                    results.append((movie, score))

            # Sort by score in descending order (stable sort preserves original order for ties)
            results.sort(key=lambda x: x[1], reverse=True)

            for movie, score in results[:args.num_results]:
                print(movie.get("title"))
        case "build":
            build_command(args.data_file, args.index_dir)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
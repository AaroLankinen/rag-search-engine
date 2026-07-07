import argparse
import json
import sys
import string

def preprocess_text(text: str) -> list[str]:
    translator = str.maketrans("", "", string.punctuation)
    clean_text = text.lower().translate(translator)
    return clean_text.split()

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

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("data_file", type=str, nargs="?", default="data/movies.json", help="Path to data file")
    search_parser.add_argument("num_results", type=int, nargs="?", default=5, help="Number of results to return")

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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
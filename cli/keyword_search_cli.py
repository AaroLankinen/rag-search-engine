import argparse
import json
import sys

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("data_file", type=str, nargs="?", default="data/movies.json", help="Path to data file")
    search_parser.add_argument("num_results", type=int, nargs="?", default=5, help="Number of results to return")

    args = parser.parse_args()

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
            query_lower = args.query.lower()
            for movie in movies:
                if not isinstance(movie, dict):
                    continue
                title = movie.get("title", "")
                if query_lower in title.lower():
                    results.append(movie)

            for movie in results[:args.num_results]:
                print(movie.get("title"))
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
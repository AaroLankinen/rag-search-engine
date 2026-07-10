import argparse
import json
import sys
try:
    from cli.lib.hybrid_search import normalize_scores,HybridSearch
except ImportError:
    from lib.hybrid_search import normalize_scores,HybridSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores to the interval 0-1")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="List of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
    weighted_search_parser.add_argument("query", help="Query to search")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weight between semantic and BM25 search (0 to 1)")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of search results to return")
    weighted_search_parser.add_argument("--data_file", nargs="?", default="data/movies.json", help="Path to the movie dataset JSON")
    weighted_search_parser.add_argument("--save_dir", nargs="?", default="cache", help="Directory containing index/embeddings")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform RRF hybrid search")
    rrf_search_parser.add_argument("query", help="Query to search")
    rrf_search_parser.add_argument("--k", type=int, default=60, help="RRF parameter")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of search results to return")
    rrf_search_parser.add_argument("--data_file", nargs="?", default="data/movies.json", help="Path to the movie dataset JSON")
    rrf_search_parser.add_argument("--save_dir", nargs="?", default="cache", help="Directory containing index/embeddings")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            print([round(s, 4) for s in normalized])
        case "weighted-search":
            try:
                with open(args.data_file, "r", encoding="utf-8") as f:
                    movies_data = json.load(f)
            except FileNotFoundError:
                print(f"Error: Data file '{args.data_file}' not found.", file=sys.stderr)
                sys.exit(1)
            except json.JSONDecodeError:
                print(f"Error: Data file '{args.data_file}' is not valid JSON.", file=sys.stderr)
                sys.exit(1)

            movies = movies_data.get("movies", []) if isinstance(movies_data, dict) else movies_data
            
            try:
                from cli.lib.hybrid_search import HybridSearch
            except ImportError:
                from lib.hybrid_search import HybridSearch

            hybrid_search = HybridSearch(movies, index_dir=args.save_dir)
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            
            for i, res in enumerate(results, start=1):
                doc = res["document"]
                title = doc.get("title", "")
                desc = doc.get("description", "")
                truncated_desc = desc[:100] + "..." if len(desc) > 100 else desc
                
                print(f"{i}. {title}")
                print(f"  Hybrid Score: {res['hybrid_score']:.3f}")
                print(f"  BM25: {res['bm25_score']:.3f}, Semantic: {res['semantic_score']:.3f}")
                print(f"  {truncated_desc}")
        case "rrf-search":
            try:
                with open(args.data_file, "r", encoding="utf-8") as f:
                    movies_data = json.load(f)
            except FileNotFoundError:
                print(f"Error: Data file '{args.data_file}' not found.", file=sys.stderr)
                sys.exit(1)
            except json.JSONDecodeError:
                print(f"Error: Data file '{args.data_file}' is not valid JSON.", file=sys.stderr)
                sys.exit(1)

            movies = movies_data.get("movies", []) if isinstance(movies_data, dict) else movies_data
            
            try:
                from cli.lib.hybrid_search import HybridSearch
            except ImportError:
                from lib.hybrid_search import HybridSearch

            hybrid_search = HybridSearch(movies, index_dir=args.save_dir)
            results = hybrid_search.rrf_search(args.query, args.k, args.limit)
            
            for i, res in enumerate(results, start=1):
                doc = res["document"]
                title = doc.get("title", "")
                desc = doc.get("description", "")
                truncated_desc = desc[:100] + "..." if len(desc) > 100 else desc
                
                print(f"{i}. {title}")
                print(f"  RRF Score: {res['rrf_score']:.3f}")
                print(f"  BM25: {res['bm25_score']:.3f}, Semantic: {res['semantic_score']:.3f}")
                print(f"  {truncated_desc}")
        case _: 
            parser.print_help()

if __name__ == "__main__":
    main()
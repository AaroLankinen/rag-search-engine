import argparse
from lib.semantic_search import SemanticSearch, verify_model, embed_text, verify_embeddings, embed_query

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the model")
    embed_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_parser.add_argument("text", help="Text to embed")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify the embeddings")
    verify_embeddings_parser.add_argument("save_dir", nargs="?", default="cache", help="Directory to save embeddings")
    embed_query_parser = subparsers.add_parser("embed_query", help="Embed a query")
    embed_query_parser.add_argument("query", help="Query to embed")
    embed_query_parser.add_argument("save_dir", nargs="?", default="cache", help="Directory to save embeddings")
    search_parser = subparsers.add_parser("search", help="Search the collection")
    search_parser.add_argument("query", help="Query to search")
    search_parser.add_argument("--data_file", nargs="?", default="data/movies.json", help="Path to the movie dataset JSON")
    search_parser.add_argument("--save_dir", nargs="?", default="cache", help="Directory to save embeddings")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of search results to return")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings(args.save_dir)
        case "embed_query":
            embed_query(args.query)
        case "search":
            import json
            with open(args.data_file, "r", encoding="utf-8") as f:
                movies_data = json.load(f)
            movies = movies_data.get("movies", []) if isinstance(movies_data, dict) else movies_data
            documents = {
                str(movie["id"]): f"{movie.get('title', '')}\n{movie.get('description', '')}"
                for movie in movies
                if "id" in movie
            }
            semantic_search = SemanticSearch()
            semantic_search.load_or_create_embeddings(documents, args.save_dir)
            results = semantic_search.search(args.query, args.limit)
            print("Search Results:")
            for doc_id in results:
                full_doc = documents.get(doc_id, "Unknown Title")
                title = full_doc.split("\n", 1)[0]
                print(f"  {title}")
        case _:
            parser.print_help()
            

if __name__ == "__main__":
    main()
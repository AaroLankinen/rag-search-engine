import argparse
try:
    from cli.lib.semantic_search import SemanticSearch, verify_model, embed_text, verify_embeddings, embed_query
except ImportError:
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
    chunk_parser = subparsers.add_parser("chunk", help="Chunk a document into smaller documents")
    chunk_parser.add_argument("text", help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Maximum number of tokens per chunk")
    chunk_parser.add_argument("--overlap", type=int, default=50, help="Number of tokens to overlap between chunks" )
    
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
            import sys
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
        case "chunk":
                chunk_document(args.text, args.chunk_size, args.overlap)
        case _:
            parser.print_help()
            
def chunk_document(text: str, max_tokens: int = 200, overlap: int = 50):
    if max_tokens <= 0:
        max_tokens = 1
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens - overlap):
        chunk = " ".join(words[i:i + max_tokens])
        chunks.append(chunk)
    
    print(f"Chunking {len(text)} characters")
    for idx, chunk in enumerate(chunks, 1):
        print(f"{idx}. {chunk}")
        

if __name__ == "__main__":
    main()
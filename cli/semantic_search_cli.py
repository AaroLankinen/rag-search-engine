import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query   

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

if __name__ == "__main__":
    main()
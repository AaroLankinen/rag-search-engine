import argparse
from lib.semantic_search import verify_model, embed_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the model")
    embed_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_parser.add_argument("text", help="Text to embed")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)

if __name__ == "__main__":
    main()
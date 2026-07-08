import argparse
from lib.semantic_search import verify_model

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the model")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

if __name__ == "__main__":
    main()
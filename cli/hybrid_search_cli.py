import argparse
try:
    from cli.lib.hybrid_search import normalize_scores
except ImportError:
    from lib.hybrid_search import normalize_scores

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores to the interval 0-1")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="List of scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            print([round(s, 4) for s in normalized])
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
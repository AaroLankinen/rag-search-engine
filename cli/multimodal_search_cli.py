import argparse
import sys

try:
    from cli.lib.multimodal_search import verify_image_embedding
except ImportError:
    from lib.multimodal_search import verify_image_embedding

def main():
    parser = argparse.ArgumentParser(description="Multimodal Image Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify_image_embedding command
    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    verify_parser.add_argument("image_path", type=str, help="Path to the image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
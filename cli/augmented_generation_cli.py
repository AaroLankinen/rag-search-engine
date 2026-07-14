import argparse
import json
import os
import sys
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

try:
    from cli.keyword_search_cli import InvertedIndex, preprocess_text
    from cli.lib.semantic_search import SemanticSearch
    from cli.lib.hybrid_search import HybridSearch
except ImportError:
    from keyword_search_cli import InvertedIndex, preprocess_text
    from lib.semantic_search import SemanticSearch
    from lib.hybrid_search import HybridSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # RAG Command
    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument("--k", type=int, default=60, help="RRF parameter")
    rag_parser.add_argument("--data_file", nargs="?", default="data/movies.json", help="Path to the movie dataset JSON")
    rag_parser.add_argument("--save_dir", nargs="?", default="cache", help="Directory containing index/embeddings")

    # Summarize Command
    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform RRF search and summarize results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Maximum number of search results to return")
    summarize_parser.add_argument("--k", type=int, default=60, help="RRF parameter")
    summarize_parser.add_argument("--data_file", nargs="?", default="data/movies.json", help="Path to the movie dataset JSON")
    summarize_parser.add_argument("--save_dir", nargs="?", default="cache", help="Directory containing index/embeddings")

    # Citations Command
    citations_parser = subparsers.add_parser(
        "citations", help="Perform RRF search and get answer with citations"
    )
    citations_parser.add_argument("query", type=str, help="Search query for citations")
    citations_parser.add_argument("--limit", type=int, default=5, help="Maximum number of search results to return")
    citations_parser.add_argument("--k", type=int, default=60, help="RRF parameter")
    citations_parser.add_argument("--data_file", nargs="?", default="data/movies.json", help="Path to the movie dataset JSON")
    citations_parser.add_argument("--save_dir", nargs="?", default="cache", help="Directory containing index/embeddings")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            
            # 1. Load movies
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

            # 2. Run RRF search using HybridSearch
            hybrid_search = HybridSearch(movies, index_dir=args.save_dir)
            results = hybrid_search.rrf_search(query, k=args.k, limit=5)

            # 3. Print retrieved Search Results titles
            print("Search Results:")
            for res in results:
                print(f"- {res['document'].get('title')}")
            print()

            # 4. Initialize LLM client
            # Resolve the absolute path to the workspace .env file
            cli_dir = os.path.dirname(os.path.abspath(__file__))
            dotenv_path = os.path.join(os.path.dirname(cli_dir), '.env')
            load_dotenv(dotenv_path, override=True)

            openrouter_key = os.environ.get("OPENROUTER_API_KEY")
            hf_token = os.environ.get("HF_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")

            if openrouter_key:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key,
                )
                model = "openrouter/free"
            elif hf_token:
                client = OpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=hf_token,
                )
                model = os.environ.get("HF_RERANK_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
            else:
                raise RuntimeError("Neither OPENROUTER_API_KEY nor HF_ACCESS_TOKEN is set in environment")

            # 5. Build RAG prompt
            context_list = []
            for res in results:
                doc = res["document"]
                context_list.append(f"Title: {doc.get('title')}\nDescription: {doc.get('description', '') or doc.get('document', '')}")
            movies_context = "\n\n".join(context_list)

            system_prompt = "You are a helpful assistant that answers questions based on a retrieved set of movies. Answer the query concisely using the provided context."
            user_prompt = f"""Query: {query}

Retrieved Movies:
{movies_context}

Please generate an answer to the query based on these movies.
"""

            # 6. Generate answer using LLM
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            answer = response.choices[0].message.content.strip()

            print("RAG Response:")
            print(answer)

        case "summarize":
            query = args.query
            
            # 1. Load movies
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

            # 2. Run RRF search using HybridSearch (ChunkedSemanticSearch / fallback unchunked)
            hybrid_search = HybridSearch(movies, index_dir=args.save_dir)
            results = hybrid_search.rrf_search(query, k=args.k, limit=args.limit)

            # 3. Print retrieved Search Results titles (indented with 2 spaces)
            print("Search Results:")
            for res in results:
                print(f"  - {res['document'].get('title')}")
            print()

            # 4. Initialize LLM client
            # Resolve the absolute path to the workspace .env file
            cli_dir = os.path.dirname(os.path.abspath(__file__))
            dotenv_path = os.path.join(os.path.dirname(cli_dir), '.env')
            load_dotenv(dotenv_path, override=True)

            openrouter_key = os.environ.get("OPENROUTER_API_KEY")
            hf_token = os.environ.get("HF_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")

            if openrouter_key:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key,
                )
                model = "openrouter/free"
            elif hf_token:
                client = OpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=hf_token,
                )
                model = os.environ.get("HF_RERANK_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
            else:
                raise RuntimeError("Neither OPENROUTER_API_KEY nor HF_ACCESS_TOKEN is set in environment")

            # 5. Build prompt
            context_list = []
            for res in results:
                doc = res["document"]
                context_list.append(f"Title: {doc.get('title')}\nDescription: {doc.get('description', '') or doc.get('document', '')}")
            movies_context = "\n\n".join(context_list)

            system_prompt = "You are a helpful assistant that summarizes search results of movies. Answer concisely based on the provided context."
            user_prompt = f"""Movies to summarize:
{movies_context}

Please provide a concise summary of these movies.
"""

            # 6. Generate summary using LLM
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            summary = response.choices[0].message.content.strip()

            print("LLM Summary:")
            print(summary)

        case "citations":
            query = args.query
            
            # 1. Load movies
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

            # 2. Run RRF search using HybridSearch
            hybrid_search = HybridSearch(movies, index_dir=args.save_dir)
            results = hybrid_search.rrf_search(query, k=args.k, limit=args.limit)

            # 3. Print retrieved Search Results titles (indented with 2 spaces)
            print("Search Results:")
            for res in results:
                print(f"  - {res['document'].get('title')}")
            print()

            # 4. Initialize LLM client
            # Resolve the absolute path to the workspace .env file
            cli_dir = os.path.dirname(os.path.abspath(__file__))
            dotenv_path = os.path.join(os.path.dirname(cli_dir), '.env')
            load_dotenv(dotenv_path, override=True)

            openrouter_key = os.environ.get("OPENROUTER_API_KEY")
            hf_token = os.environ.get("HF_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")

            if openrouter_key:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key,
                )
                model = "openrouter/free"
            elif hf_token:
                client = OpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=hf_token,
                )
                model = os.environ.get("HF_RERANK_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
            else:
                raise RuntimeError("Neither OPENROUTER_API_KEY nor HF_ACCESS_TOKEN is set in environment")

            # 5. Build prompt with cited documents
            doc_strings = []
            for i, res in enumerate(results, 1):
                doc = res["document"]
                doc_strings.append(f"[{i}] Title: {doc.get('title')}\nDescription: {doc.get('description', '') or doc.get('document', '')}")
            documents_str = "\n\n".join(doc_strings)

            user_prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Hoopla, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{documents_str}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

            # 6. Generate answer using LLM
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            answer = response.choices[0].message.content.strip()

            print("LLM Answer:")
            print(answer)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
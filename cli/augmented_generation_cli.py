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
except ImportError:
    from keyword_search_cli import InvertedIndex, preprocess_text
    from lib.semantic_search import SemanticSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument("--k", type=int, default=60, help="RRF parameter")
    rag_parser.add_argument("--data_file", nargs="?", default="data/movies.json", help="Path to the movie dataset JSON")
    rag_parser.add_argument("--save_dir", nargs="?", default="cache", help="Directory containing index/embeddings")

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
            movie_map = {str(m["id"]): m for m in movies}

            # 2. Run RRF search using standard SemanticSearch
            # Load BM25 Inverted Index
            idx = InvertedIndex([])
            idx.load(args.save_dir)

            # Load Semantic Search embeddings
            semantic_search = SemanticSearch()
            semantic_search.load_or_create_embeddings(idx.doc_map, args.save_dir)

            # Run BM25 search
            bm25_scores = {}
            query_tokens = preprocess_text(query)
            for token in query_tokens:
                for doc_id in idx.get_documents(token):
                    bm25_scores[doc_id] = idx.bm25(doc_id, query)

            sorted_bm25 = sorted(bm25_scores.items(), key=lambda x: (x[1], -int(x[0])), reverse=True)
            bm25_ranks = {doc_id: i + 1 for i, (doc_id, _) in enumerate(sorted_bm25)}

            # Run Semantic Search logic
            query_embedding = semantic_search.generate_embedding(query)
            q_norm_val = np.linalg.norm(query_embedding)
            q_norm = query_embedding / q_norm_val if q_norm_val > 0 else query_embedding

            norms = np.linalg.norm(semantic_search.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb_normalized = semantic_search.embeddings / norms
            similarities = np.dot(emb_normalized, q_norm)

            sorted_indices = np.argsort(similarities)[::-1]
            semantic_ranks = {semantic_search.document_map[idx_val]: i + 1 for i, idx_val in enumerate(sorted_indices)}

            # Hybrid / RRF
            all_docs = set(bm25_ranks.keys()) | set(semantic_ranks.keys())
            rrf_scores = {}
            k_const = args.k
            for doc_id in all_docs:
                score = 0.0
                if doc_id in bm25_ranks:
                    score += 1.0 / (k_const + bm25_ranks[doc_id])
                if doc_id in semantic_ranks:
                    score += 1.0 / (k_const + semantic_ranks[doc_id])
                rrf_scores[doc_id] = score

            sorted_rrf = sorted(rrf_scores.items(), key=lambda x: (x[1], -int(x[0])), reverse=True)
            top_results_ids = [doc_id for doc_id, _ in sorted_rrf[:5]]

            # 3. Print retrieved Search Results titles
            print("Search Results:")
            for doc_id in top_results_ids:
                m = movie_map[doc_id]
                print(f"- {m.get('title')}")
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
            for doc_id in top_results_ids:
                m = movie_map[doc_id]
                context_list.append(f"Title: {m.get('title')}\nDescription: {m.get('description', '') or m.get('document', '')}")
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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
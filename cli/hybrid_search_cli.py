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
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual"], help="Reranking method to use")

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

            query = args.query
            if args.enhance in ["spell", "rewrite", "expand"]:
                import os
                from dotenv import load_dotenv
                from openai import OpenAI

                # Resolve the absolute path to the workspace .env file
                cli_dir = os.path.dirname(os.path.abspath(__file__))
                dotenv_path = os.path.join(os.path.dirname(cli_dir), '.env')
                load_dotenv(dotenv_path, override=True)

                api_key = os.environ.get("OPENROUTER_API_KEY")
                if not api_key:
                    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )

                if args.enhance == "spell":
                    system_prompt = "You are a spelling correction assistant. Correct any spelling or typographical errors in the user query. Do not add any conversational text, explanations, or quotes. Respond ONLY with the corrected query."
                elif args.enhance == "rewrite":
                    system_prompt = "You are a query optimization assistant. Rewrite the user query into a concise, Google-style keyword search query designed to yield highly relevant search results. Do not add any conversational text, explanations, or quotes. Respond ONLY with the rewritten query."
                else:
                    system_prompt = "You are a query expansion assistant. Expand the user query by appending synonyms, related concepts, and broader search terms to improve recall. Do not add any conversational text, explanations, or quotes. Respond ONLY with the expanded query."

                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                )
                enhanced_query = response.choices[0].message.content.strip().strip('"').strip("'")
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n", end="")
                if enhanced_query:
                    query = enhanced_query

            hybrid_search = HybridSearch(movies, index_dir=args.save_dir)
            
            if getattr(args, "rerank_method", None) == "individual":
                import os
                import time
                import re
                from dotenv import load_dotenv
                from openai import OpenAI

                # Resolve the absolute path to the workspace .env file
                cli_dir = os.path.dirname(os.path.abspath(__file__))
                dotenv_path = os.path.join(os.path.dirname(cli_dir), '.env')
                load_dotenv(dotenv_path, override=True)

                hf_token = os.environ.get("HF_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")
                if not hf_token:
                    raise RuntimeError("HF_ACCESS_TOKEN environment variable not set")

                model = os.environ.get("HF_RERANK_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

                try:
                    test_client = OpenAI(
                        base_url="https://router.huggingface.co/v1",
                        api_key=hf_token,
                    )
                    # Verify permission with a quick call
                    test_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "ping"}],
                        max_tokens=1,
                    )
                    client = test_client
                except Exception:
                    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
                    if openrouter_key:
                        client = OpenAI(
                            base_url="https://openrouter.ai/api/v1",
                            api_key=openrouter_key,
                        )
                        model = "meta-llama/llama-3.3-70b-instruct"
                    else:
                        client = OpenAI(
                            base_url="https://router.huggingface.co/v1",
                            api_key=hf_token,
                        )

                results = hybrid_search.rrf_search(query, args.k, args.limit * 5)
                
                print(f"Re-ranking top {args.limit} results using individual method...")
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")

                reranked_results = []
                for res in results:
                    doc = res["document"]
                    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "") or doc.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""

                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.0,
                            max_tokens=5,
                        )
                        raw_score = response.choices[0].message.content.strip()
                        match = re.search(r"(\d+(\.\d+)?)", raw_score)
                        if match:
                            score = float(match.group(1))
                        else:
                            score = 0.0
                    except Exception:
                        score = 0.0

                    res["re_rank_score"] = min(max(score, 0.0), 10.0)
                    reranked_results.append(res)
                    time.sleep(0.2)

                # Sort by re_rank_score descending, then rrf_score descending, then document ID ascending
                sorted_reranked = sorted(
                    reranked_results,
                    key=lambda x: (x["re_rank_score"], x["rrf_score"], -int(x["document"]["id"])),
                    reverse=True,
                )

                for i, res in enumerate(sorted_reranked[:args.limit], start=1):
                    doc = res["document"]
                    title = doc.get("title", "")
                    desc = doc.get("description", "")
                    truncated_desc = desc[:100] + "..." if len(desc) > 100 else desc
                    
                    bm25_rank = res.get("bm25_rank")
                    sem_rank = res.get("semantic_rank")
                    
                    bm25_rank_str = str(bm25_rank) if bm25_rank is not None else "N/A"
                    sem_rank_str = str(sem_rank) if sem_rank is not None else "N/A"
                    
                    print(f"{i}. {title}")
                    print(f"   Re-rank Score: {res['re_rank_score']:.3f}/10")
                    print(f"   RRF Score: {res['rrf_score']:.3f}")
                    print(f"   BM25 Rank: {bm25_rank_str}, Semantic Rank: {sem_rank_str}")
                    print(f"   {truncated_desc}")
                    print()
            else:
                results = hybrid_search.rrf_search(query, args.k, args.limit)
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
import argparse
import json
import sys
import logging
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
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Reranking method to use")
    rrf_search_parser.add_argument("--debug", action="store_true", help="Enable comprehensive debug logging")
    rrf_search_parser.add_argument("--evaluate", action="store_true", help="Evaluate search results using an LLM on a 0-3 scale")

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
            if getattr(args, "debug", False):
                logging.basicConfig(
                    level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    stream=sys.stderr
                )
            logger = logging.getLogger("hybrid_search_cli")
            logger.debug("Debug logging enabled for RRF Search CLI.")

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

                final_results = sorted_reranked[:args.limit]
                if not getattr(args, "evaluate", False):
                    for i, res in enumerate(final_results, start=1):
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
            elif getattr(args, "rerank_method", None) == "batch":
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

                print(f"Re-ranking top {args.limit} results using batch method...")
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")

                # Build the document list string for the batch prompt
                doc_list_str = ""
                result_by_id = {}
                for res in results:
                    doc = res["document"]
                    doc_id = int(doc["id"])
                    title = doc.get("title", "")
                    desc = doc.get("description", "") or doc.get("document", "")
                    truncated_desc = desc[:200] + "..." if len(desc) > 200 else desc
                    doc_list_str += f"ID: {doc_id} | Title: {title} | Description: {truncated_desc}\n"
                    result_by_id[doc_id] = res

                prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}
Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

                ranked_ids = None
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.0,
                            max_tokens=512,
                        )
                        raw_text = response.choices[0].message.content.strip()
                        # Try to extract a JSON array from the response
                        # Find the first '[' and last ']' to handle any wrapping text
                        bracket_start = raw_text.find('[')
                        bracket_end = raw_text.rfind(']')
                        if bracket_start != -1 and bracket_end != -1:
                            json_str = raw_text[bracket_start:bracket_end + 1]
                            parsed = json.loads(json_str)
                            if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
                                ranked_ids = parsed
                                break
                        # If parsing failed, retry
                    except Exception:
                        if attempt < max_retries - 1:
                            time.sleep(1)
                        continue

                if ranked_ids is None:
                    # Fallback: use the original RRF order
                    ranked_ids = [int(res["document"]["id"]) for res in results]

                # Build the output list in the order returned by the LLM
                seen = set()
                ordered_results = []
                for doc_id in ranked_ids:
                    if doc_id in result_by_id and doc_id not in seen:
                        seen.add(doc_id)
                        ordered_results.append(result_by_id[doc_id])

                # Append any results not mentioned by the LLM at the end (in original order)
                for res in results:
                    doc_id = int(res["document"]["id"])
                    if doc_id not in seen:
                        seen.add(doc_id)
                        ordered_results.append(res)

                final_results = ordered_results[:args.limit]
                if not getattr(args, "evaluate", False):
                    for i, res in enumerate(final_results, start=1):
                        doc = res["document"]
                        title = doc.get("title", "")
                        desc = doc.get("description", "")
                        truncated_desc = desc[:100] + "..." if len(desc) > 100 else desc

                        bm25_rank = res.get("bm25_rank")
                        sem_rank = res.get("semantic_rank")

                        bm25_rank_str = str(bm25_rank) if bm25_rank is not None else "N/A"
                        sem_rank_str = str(sem_rank) if sem_rank is not None else "N/A"

                        print(f"{i}. {title}")
                        print(f"   Re-rank Rank: {i}")
                        print(f"   RRF Score: {res['rrf_score']:.3f}")
                        print(f"   BM25 Rank: {bm25_rank_str}, Semantic Rank: {sem_rank_str}")
                        print(f"   {truncated_desc}")
                        print()
            elif getattr(args, "rerank_method", None) == "cross_encoder":
                from sentence_transformers import CrossEncoder

                logger.debug("Requesting RRF search results. k: %d, limit for reranking: %d", args.k, args.limit * 5)
                results = hybrid_search.rrf_search(query, args.k, args.limit * 5)
                logger.debug("RRF search returned %d candidate documents for cross-encoder reranking.", len(results))

                print(f"Re-ranking top {args.limit} results using cross_encoder method...")
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")

                # Build query-document pairs for the CrossEncoder
                pairs = []
                for res in results:
                    doc = res["document"]
                    doc_text = f"{doc.get('title', '')} - {doc.get('description', '') or doc.get('document', '')}"
                    pairs.append((query, doc_text))

                logger.debug("Loading CrossEncoder model 'cross-encoder/ms-marco-MiniLM-L-6-v2'...")
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.debug("Computing cross-encoder scores for %d candidate pairs...", len(pairs))
                scores = cross_encoder.predict(pairs)

                # Attach scores to results
                for res, score in zip(results, scores):
                    res["cross_encoder_score"] = float(score)

                # Sort by cross_encoder_score descending, then rrf_score descending,
                # then document ID ascending
                sorted_results = sorted(
                    results,
                    key=lambda x: (x["cross_encoder_score"], x["rrf_score"], -int(x["document"]["id"])),
                    reverse=True,
                )

                logger.debug("CrossEncoder reranking completed. Scores for top candidates:")
                for rank, res in enumerate(sorted_results, start=1):
                    doc = res["document"]
                    logger.debug(
                        "  Rerank %d: %s (ID: %s) -> CrossEncoder Score: %.4f, RRF Score: %.6f",
                        rank,
                        doc.get("title", ""),
                        doc.get("id"),
                        res["cross_encoder_score"],
                        res["rrf_score"],
                    )

                # Log details about why "The Land Before Time XI" didn't show up in top results
                target_title = "The Land Before Time XI: Invasion of the Tinysauruses"
                target_res = next((res for res in sorted_results if res["document"].get("title") == target_title), None)
                if target_res:
                    target_rank = sorted_results.index(target_res) + 1
                    logger.debug(
                        "RERANK TARGET INFO - '%s' (ID: %s) was reranked as rank %d out of %d candidates. CrossEncoder Score: %.4f (vs 5th-place score: %.4f)",
                        target_title,
                        target_res["document"].get("id"),
                        target_rank,
                        len(sorted_results),
                        target_res["cross_encoder_score"],
                        sorted_results[4]["cross_encoder_score"] if len(sorted_results) >= 5 else 0.0,
                    )
                else:
                    logger.debug(
                        "RERANK TARGET INFO - '%s' was NOT in the top %d candidate pool retrieved by RRF search.",
                        target_title,
                        args.limit * 5,
                    )

                final_results = sorted_results[:args.limit]
                if not getattr(args, "evaluate", False):
                    for i, res in enumerate(final_results, start=1):
                        doc = res["document"]
                        title = doc.get("title", "")
                        desc = doc.get("description", "")
                        truncated_desc = desc[:100] + "..." if len(desc) > 100 else desc

                        bm25_rank = res.get("bm25_rank")
                        sem_rank = res.get("semantic_rank")

                        bm25_rank_str = str(bm25_rank) if bm25_rank is not None else "N/A"
                        sem_rank_str = str(sem_rank) if sem_rank is not None else "N/A"

                        print(f"{i}. {title}")
                        print(f"   Cross Encoder Score: {res['cross_encoder_score']:.3f}")
                        print(f"   RRF Score: {res['rrf_score']:.3f}")
                        print(f"   BM25 Rank: {bm25_rank_str}, Semantic Rank: {sem_rank_str}")
                        print(f"   {truncated_desc}")
                        print()
            else:
                results = hybrid_search.rrf_search(query, args.k, args.limit)
                final_results = results
                if not getattr(args, "evaluate", False):
                    for i, res in enumerate(final_results, start=1):
                        doc = res["document"]
                        title = doc.get("title", "")
                        desc = doc.get("description", "")
                        truncated_desc = desc[:100] + "..." if len(desc) > 100 else desc
                        
                        print(f"{i}. {title}")
                        print(f"  RRF Score: {res['rrf_score']:.3f}")
                        print(f"  BM25: {res['bm25_score']:.3f}, Semantic: {res['semantic_score']:.3f}")
                        print(f"  {truncated_desc}")

            if getattr(args, "evaluate", False):
                import os
                import time
                from dotenv import load_dotenv
                from openai import OpenAI

                # Resolve the absolute path to the workspace .env file
                cli_dir = os.path.dirname(os.path.abspath(__file__))
                dotenv_path = os.path.join(os.path.dirname(cli_dir), '.env')
                load_dotenv(dotenv_path, override=True)

                openrouter_key = os.environ.get("OPENROUTER_API_KEY")
                hf_token = os.environ.get("HF_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")

                if openrouter_key:
                    eval_client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=openrouter_key,
                    )
                    eval_model = "openrouter/free"
                elif hf_token:
                    eval_client = OpenAI(
                        base_url="https://router.huggingface.co/v1",
                        api_key=hf_token,
                    )
                    eval_model = os.environ.get("HF_RERANK_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
                else:
                    raise RuntimeError("Neither OPENROUTER_API_KEY nor HF_ACCESS_TOKEN is set in environment")

                # Build the prompt with movie titles and descriptions mapped to their IDs
                movie_details = []
                for res in final_results:
                    doc = res["document"]
                    movie_details.append(
                        f"ID: {doc['id']} | Title: {doc.get('title', '')} | Description: {doc.get('description', '') or doc.get('document', '')}"
                    )
                movie_list_str = "\n".join(movie_details)

                prompt = f"""You are an expert search quality evaluator.
Evaluate the relevance of the following search results for the query: "{query}".

Scale:
3: Highly relevant (exactly what the user is looking for, or directly addresses the query)
2: Relevant (related to the query, a good match)
1: Marginally relevant (tangentially related, but not a strong match)
0: Not relevant (unrelated to the query)

Movies to evaluate:
{movie_list_str}

Your response must be a JSON object mapping each movie ID (as a string) to its integer relevance score (0, 1, 2, or 3).
Do not wrap the response in Markdown. Do not include any explanations or other text.
Example response:
{{
  "123": 2,
  "456": 2,
  "789": 0
}}

Relevance Mapping:"""

                scores_map = {}
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = eval_client.chat.completions.create(
                            model=eval_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,
                            max_tokens=256,
                        )
                        raw_text = response.choices[0].message.content.strip()
                        
                        # Strip any markdown JSON wrapper
                        if raw_text.startswith("```"):
                            lines = raw_text.split("\n")
                            if lines[0].startswith("```"):
                                lines = lines[1:]
                            if lines[-1].startswith("```"):
                                lines = lines[:-1]
                            raw_text = "\n".join(lines).strip()
                            
                        parsed = json.loads(raw_text)
                        if isinstance(parsed, dict):
                            for k_id, score in parsed.items():
                                scores_map[str(k_id)] = int(score)
                            break
                    except Exception:
                        if attempt < max_retries - 1:
                            time.sleep(1)
                        continue

                # Print final results in the requested format
                for i, res in enumerate(final_results, start=1):
                    doc = res["document"]
                    d_id = str(doc["id"])
                    title = doc.get("title", "")
                    score = scores_map.get(d_id, 0)
                    score = min(max(score, 0), 3)
                    print(f"{i}. {title}: {score}/3")
        case _: 
            parser.print_help()

if __name__ == "__main__":
    main()
import argparse
import json
import os
import sys
import numpy as np

# Limit threads to prevent CPU deadlocks in PyTorch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
torch.set_num_threads(1)

try:
    from cli.keyword_search_cli import InvertedIndex, preprocess_text
    from cli.lib.semantic_search import SemanticSearch
except ImportError:
    from keyword_search_cli import InvertedIndex, preprocess_text
    from lib.semantic_search import SemanticSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/golden_dataset.json",
        help="Path to the dataset file",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/index",
        help="Path to the index directory",
    )
    parser.add_argument(
        "--rerank-method",
        type=str,
        default="cross_encoder",
        help="Method to use for reranking (cross_encoder, cohere, etc.)",
    )
    parser.add_argument(
        "--search-method",
        type=str,
        default="hybrid",
        help="Method to use for search (bm25, semantic, hybrid)",
    )

    args = parser.parse_args()
    limit = args.limit
    search_method = args.search_method
    dataset_path = args.dataset
    index_dir = args.index_dir

    # Check index dir and fall back to cache if data/index is empty or does not exist
    if not os.path.exists(os.path.join(index_dir, "index.pkl")) and os.path.exists(os.path.join("cache", "index.pkl")):
        index_dir = "cache"

    # Load golden dataset
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file '{dataset_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Dataset file '{dataset_path}' is not valid JSON.", file=sys.stderr)
        sys.exit(1)

    test_cases = dataset.get("test_cases", [])

    # Load BM25 Inverted Index
    inverted_index = InvertedIndex([])
    inverted_index.load(index_dir)

    # Load Semantic Search embeddings
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(inverted_index.doc_map, index_dir)

    print(f"k={limit}")
    print()

    for idx, test_case in enumerate(test_cases):
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]

        # Run BM25 search logic (score all matching docs)
        bm25_scores = {}
        query_tokens = preprocess_text(query)
        for token in query_tokens:
            for doc_id in inverted_index.get_documents(token):
                bm25_scores[doc_id] = inverted_index.bm25(doc_id, query)
        
        sorted_bm25 = sorted(bm25_scores.items(), key=lambda x: (x[1], -int(x[0])), reverse=True)
        bm25_ranks = {doc_id: i + 1 for i, (doc_id, _) in enumerate(sorted_bm25)}

        # Run Semantic Search logic (score all docs using numpy cosine similarity)
        query_embedding = semantic_search.generate_embedding(query)
        q_norm_val = np.linalg.norm(query_embedding)
        q_norm = query_embedding / q_norm_val if q_norm_val > 0 else query_embedding

        norms = np.linalg.norm(semantic_search.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_normalized = semantic_search.embeddings / norms
        similarities = np.dot(emb_normalized, q_norm)

        sorted_indices = np.argsort(similarities)[::-1]
        semantic_ranks = {semantic_search.document_map[idx]: i + 1 for i, idx in enumerate(sorted_indices)}

        # Select retrieval list based on search method
        if search_method == "bm25":
            retrieved_ids = [doc_id for doc_id, _ in sorted_bm25[:limit]]
        elif search_method == "semantic":
            retrieved_ids = [semantic_search.document_map[idx] for idx in sorted_indices[:limit]]
        else:
            # Hybrid / RRF
            all_docs = set(bm25_ranks.keys()) | set(semantic_ranks.keys())
            rrf_scores = {}
            k_const = 60
            for doc_id in all_docs:
                score = 0.0
                if doc_id in bm25_ranks:
                    score += 1.0 / (k_const + bm25_ranks[doc_id])
                if doc_id in semantic_ranks:
                    score += 1.0 / (k_const + semantic_ranks[doc_id])
                rrf_scores[doc_id] = score
            
            sorted_rrf = sorted(rrf_scores.items(), key=lambda x: (x[1], -int(x[0])), reverse=True)
            retrieved_ids = [doc_id for doc_id, _ in sorted_rrf[:limit]]

        # Map IDs to titles
        retrieved_titles = []
        for doc_id in retrieved_ids:
            full_doc = inverted_index.doc_map.get(doc_id, "Unknown Title")
            title = full_doc.split("\n", 1)[0]
            retrieved_titles.append(title)

        # Calculate Precision@limit
        relevant_set = set(relevant_docs)
        relevant_retrieved = sum(1 for title in retrieved_titles if title in relevant_set)
        precision = relevant_retrieved / limit if limit > 0 else 0.0

        # Print formatting
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_docs)}")

        if idx < len(test_cases) - 1:
            print()

if __name__ == "__main__":
    main()
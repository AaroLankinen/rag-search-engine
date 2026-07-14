import sys
sys.path.insert(0, "cli")
import json
import numpy as np
from keyword_search_cli import InvertedIndex, preprocess_text
from lib.semantic_search import ChunkedSemanticSearch

with open("data/movies.json", "r", encoding="utf-8") as f:
    movies_data = json.load(f)
movies = movies_data.get("movies", []) if isinstance(movies_data, dict) else movies_data
movie_map = {str(m["id"]): m for m in movies}

idx = InvertedIndex([])
idx.load("cache")

chunked = ChunkedSemanticSearch()
documents_dict = {
    str(doc["id"]): f"{doc.get('title', '')}\n{doc.get('description', '')}"
    for doc in movies
    if "id" in doc
}
chunked.load_or_create_chunk_embeddings(documents_dict, "cache")

query = "action movie with lasers"

# BM25 scores
bm25_scores = {}
query_tokens = preprocess_text(query)
for token in query_tokens:
    for doc_id in idx.get_documents(token):
        bm25_scores[doc_id] = idx.bm25(doc_id, query)
sorted_bm25 = sorted(bm25_scores.items(), key=lambda x: (x[1], -int(x[0])), reverse=True)
bm25_ranks = {doc_id: i + 1 for i, (doc_id, _) in enumerate(sorted_bm25)}

# Semantic search scores
chunk_results = chunked.search(query, limit=len(chunked.chunk_embeddings))
semantic_scores = {}
for res in chunk_results:
    doc_id = res["doc_id"]
    score = res["score"]
    if doc_id not in semantic_scores or score > semantic_scores[doc_id]:
        semantic_scores[doc_id] = score
sorted_sem = sorted(semantic_scores.items(), key=lambda x: (x[1], -int(x[0])), reverse=True)
semantic_ranks = {doc_id: i + 1 for i, (doc_id, _) in enumerate(sorted_sem)}

# RRF fusion
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

# Find Eliminators
elim_rank = None
for rank, (doc_id, score) in enumerate(sorted_rrf, 1):
    if doc_id == "638":
         elim_rank = rank
         print(f"Eliminators Chunked RRF Rank: {rank}, Score: {score:.5f}, BM25 rank: {bm25_ranks.get(doc_id)}, Semantic rank: {semantic_ranks.get(doc_id)}")
         break
else:
     print("Eliminators not found in candidates!")

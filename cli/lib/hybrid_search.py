import os

try:
    from cli.keyword_search_cli import InvertedIndex
except ImportError:
    try:
        from ..keyword_search_cli import InvertedIndex
    except ImportError:
        from keyword_search_cli import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents: list[dict], index_dir: str = "cache") -> None:
        self.documents = documents
        self.index_dir = index_dir

        # Convert documents list[dict] to dict[str, str] for keyword/semantic search
        documents_dict = {
            str(doc["id"]): f"{doc.get('title', '')}\n{doc.get('description', '')}"
            for doc in documents
            if "id" in doc
        }

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents_dict, save_dir=index_dir)

        self.idx = InvertedIndex([])
        index_file = os.path.join(index_dir, "index.pkl")
        if not os.path.exists(index_file):
            self.idx = InvertedIndex(documents_dict)
            self.idx.save(index_dir)
        else:
            self.idx.load(index_dir)

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load(self.index_dir)
        doc_ids = self.idx.bm25_search(query, limit)
        doc_map = {str(d["id"]): d for d in self.documents}
        return [doc_map[d_id] for d_id in doc_ids if d_id in doc_map]

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        """Perform weighted hybrid search combining semantic and BM25 scores.

        Uses the top semantic search results as the candidate pool, computes
        BM25 scores for each candidate, normalizes both score types within the
        pool, and combines them using the alpha weight.

        Args:
            query: The search query string.
            alpha: Weight for semantic scores (0 to 1). BM25 weight is (1 - alpha).
            limit: Maximum number of results to return.

        Returns:
            A list of result dicts, each containing 'document', 'hybrid_score',
            'bm25_score', and 'semantic_score'.
        """
        # 1. Get raw semantic scores for all chunks, aggregate max score per document
        chunk_results = self.semantic_search.search(
            query, limit=len(self.semantic_search.chunk_embeddings)
        )

        semantic_raw = {}
        for res in chunk_results:
            d_id = str(res["doc_id"])
            score = res["score"]
            if d_id not in semantic_raw or score > semantic_raw[d_id]:
                semantic_raw[d_id] = score

        # 2. Select the top `limit` documents by semantic score as the candidate pool
        sorted_by_sem = sorted(semantic_raw.items(), key=lambda x: x[1], reverse=True)
        candidate_ids = [d_id for d_id, _ in sorted_by_sem[:limit]]

        # 3. Compute BM25 scores for each candidate
        self.idx.load(self.index_dir)
        bm25_raw = {d_id: self.idx.bm25(d_id, query) for d_id in candidate_ids}

        # 4. Normalize both score types within the candidate pool
        raw_sem_list = [semantic_raw[d_id] for d_id in candidate_ids]
        raw_bm25_list = [bm25_raw[d_id] for d_id in candidate_ids]

        norm_sem_list = normalize_scores(raw_sem_list)
        norm_bm25_list = normalize_scores(raw_bm25_list)

        norm_sem = dict(zip(candidate_ids, norm_sem_list))
        norm_bm25 = dict(zip(candidate_ids, norm_bm25_list))

        # 5. Compute hybrid score for each candidate
        doc_map = {str(doc["id"]): doc for doc in self.documents}
        results = []
        for d_id in candidate_ids:
            b_score = norm_bm25[d_id]
            s_score = norm_sem[d_id]
            hybrid_score = alpha * s_score + (1 - alpha) * b_score
            results.append({
                "document": doc_map[d_id],
                "hybrid_score": hybrid_score,
                "bm25_score": b_score,
                "semantic_score": s_score,
            })

        # 6. Sort by hybrid_score descending; ties broken by document ID ascending
        sorted_results = sorted(
            results,
            key=lambda x: (x["hybrid_score"], -int(x["document"]["id"])),
            reverse=True,
        )
        return sorted_results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        """Perform Reciprocal Rank Fusion (RRF) hybrid search.

        Retrieves ranked lists from both semantic and BM25 searches (top 500
        for each), then combines them using the RRF formula:
            RRF(d) = 1/(k + rank_semantic(d)) + 1/(k + rank_bm25(d))
        where non-retrieved documents get a reciprocal rank contribution of 0.

        Args:
            query: The search query string.
            k: RRF smoothing constant (typically 60).
            limit: Maximum number of results to return.

        Returns:
            A list of result dicts, each containing 'document', 'rrf_score',
            'bm25_score', and 'semantic_score'.
        """
        # 1. Get semantic scores for all chunks, aggregate max per document
        chunk_results = self.semantic_search.search(
            query, limit=len(self.semantic_search.chunk_embeddings)
        )

        semantic_raw = {}
        for res in chunk_results:
            d_id = str(res["doc_id"])
            score = res["score"]
            if d_id not in semantic_raw or score > semantic_raw[d_id]:
                semantic_raw[d_id] = score

        # 2. Get top 500 semantic documents
        sem_ranked = sorted(
            semantic_raw.items(),
            key=lambda x: (x[1], -int(x[0])),
            reverse=True,
        )
        sem_limit = 500
        sem_top = sem_ranked[:sem_limit]
        sem_rank = {d_id: rank for rank, (d_id, _) in enumerate(sem_top, 1)}

        # 3. Get top 500 BM25 documents
        self.idx.load(self.index_dir)
        bm25_limit = 500
        bm25_top_ids = self.idx.bm25_search(query, limit=bm25_limit)
        bm25_rank = {d_id: rank for rank, d_id in enumerate(bm25_top_ids, 1)}

        # 4. Candidate pool is the union of top lists
        candidate_ids = list(set(sem_rank.keys()) | set(bm25_rank.keys()))

        # Compute raw scores for normalization/display
        bm25_raw = {d_id: self.idx.bm25(d_id, query) for d_id in candidate_ids}
        raw_sem_list = [semantic_raw.get(d_id, 0.0) for d_id in candidate_ids]
        raw_bm25_list = [bm25_raw[d_id] for d_id in candidate_ids]

        norm_sem_list = normalize_scores(raw_sem_list)
        norm_bm25_list = normalize_scores(raw_bm25_list)

        norm_sem = dict(zip(candidate_ids, norm_sem_list))
        norm_bm25 = dict(zip(candidate_ids, norm_bm25_list))

        # 5. Compute RRF score for each candidate
        doc_map = {str(doc["id"]): doc for doc in self.documents}
        results = []
        for d_id in candidate_ids:
            s_r = sem_rank.get(d_id)
            b_r = bm25_rank.get(d_id)

            s_term = 1 / (k + s_r) if s_r is not None else 0.0
            b_term = 1 / (k + b_r) if b_r is not None else 0.0
            rrf_score = s_term + b_term

            results.append({
                "document": doc_map[d_id],
                "rrf_score": rrf_score,
                "bm25_score": norm_bm25[d_id],
                "semantic_score": norm_sem[d_id],
            })

        # 6. Sort by rrf_score descending; ties broken by document ID ascending
        sorted_results = sorted(
            results,
            key=lambda x: (x["rrf_score"], -int(x["document"]["id"])),
            reverse=True,
        )
        return sorted_results[:limit]

def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 if max_score != 0 else 0.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]
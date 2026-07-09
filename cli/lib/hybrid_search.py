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
        # 1. Get raw semantic scores for all chunks, and find the maximum score for each document
        chunk_results = self.semantic_search.search(query, limit=len(self.semantic_search.chunk_embeddings))
        
        semantic_raw = {}
        for res in chunk_results:
            d_id = str(res["doc_id"])
            score = res["score"]
            if d_id not in semantic_raw or score > semantic_raw[d_id]:
                semantic_raw[d_id] = score
                
        # 2. Get raw BM25 scores for all documents in the index
        self.idx.load(self.index_dir)
        
        from keyword_search_cli import preprocess_text
        query_tokens = preprocess_text(query)
        candidate_doc_ids = set()
        for token in query_tokens:
            candidate_doc_ids.update(self.idx.index.get(token, []))
            
        bm25_raw = {}
        for doc in self.documents:
            d_id = str(doc["id"])
            if d_id in candidate_doc_ids:
                bm25_raw[d_id] = self.idx.bm25(d_id, query)
            else:
                bm25_raw[d_id] = 0.0
                
        # Fill in semantic scores for any document that didn't have chunk results
        for doc in self.documents:
            d_id = str(doc["id"])
            if d_id not in semantic_raw:
                semantic_raw[d_id] = 0.0

        # Extract list of raw scores to normalize
        doc_ids = [str(doc["id"]) for doc in self.documents]
        raw_bm25_list = [bm25_raw[d_id] for d_id in doc_ids]
        raw_sem_list = [semantic_raw[d_id] for d_id in doc_ids]
        
        # Normalize the raw scores using the normalize_scores function
        norm_bm25_list = normalize_scores(raw_bm25_list)
        norm_sem_list = normalize_scores(raw_sem_list)
        
        norm_bm25 = dict(zip(doc_ids, norm_bm25_list))
        norm_sem = dict(zip(doc_ids, norm_sem_list))
        
        # Compute hybrid score for each document
        results = []
        for doc in self.documents:
            d_id = str(doc["id"])
            b_score = norm_bm25[d_id]
            s_score = norm_sem[d_id]
            hybrid_score = alpha * s_score + (1 - alpha) * b_score
            results.append({
                "document": doc,
                "hybrid_score": hybrid_score,
                "bm25_score": b_score,
                "semantic_score": s_score
            })
            
        # Sort by hybrid_score descending, and in case of ties, sort by document ID ascending (represented as integer)
        sorted_results = sorted(
            results,
            key=lambda x: (x["hybrid_score"], -int(x["document"]["id"])),
            reverse=True
        )
        return sorted_results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 if max_score != 0 else 0.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]
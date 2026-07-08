import os
import torch
from sentence_transformers import SentenceTransformer, util

class SemanticSearch:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)

    def generate_embedding(self, text):
        return self.model.encode(text, convert_to_numpy=True)

    def search(self, query: str, limit: int = 10) -> list[str]:
        query_embedding = self.generate_embedding(query)
        similarities = util.cos_sim(query_embedding, self.document_embeddings)[0]
        top_k = torch.topk(similarities, k=limit)
        results = []
        for score, idx in zip(top_k.values, top_k.indices):
            results.append(list(self.documents.keys())[idx.item()])
        return results
    

def verify_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")

def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def semantic_search(query: str, limit: int = 10):
    semantic_search = SemanticSearch()
    return semantic_search.search(query, limit)
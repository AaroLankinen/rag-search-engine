import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

class SemanticSearch:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def build_embeddings(self, documents, save_dir: str = "cache"):
        import pickle
        self.documents = documents
        self.embeddings = self.model.encode(list(documents.values()), convert_to_numpy=True, show_progress_bar=True)
        self.document_map = {i: doc for i, doc in enumerate(documents.keys())}
        np.save(os.path.join(save_dir, "embeddings.npy"), self.embeddings)
        with open(os.path.join(save_dir, "document_map.pkl"), "wb") as f:
            pickle.dump(self.document_map, f)
        return self.embeddings

    def load_or_create_embeddings(self, documents, save_dir: str = "cache"):
        import pickle
        try:
            self.embeddings = np.load(os.path.join(save_dir, "embeddings.npy"))
            with open(os.path.join(save_dir, "document_map.pkl"), "rb") as f:
                self.document_map = pickle.load(f)
        except FileNotFoundError:
            self.build_embeddings(documents, save_dir)

    def generate_embedding(self, text):
        return self.model.encode(text, convert_to_numpy=True)

    def search(self, query: str, limit: int = 10) -> list[str]:
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        query_embedding = self.generate_embedding(query)
        similarities = util.cos_sim(query_embedding, self.embeddings)[0]
        k = min(limit, len(self.embeddings))
        top_k = torch.topk(similarities, k=k)
        results = []
        for score, idx in zip(top_k.values, top_k.indices):
            results.append(self.document_map[idx.item()])
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

def embed_query(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def semantic_search(query: str, limit: int = 10):
    semantic_search = SemanticSearch()
    import json
    data_file = "data/movies.json"
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            movies_data = json.load(f)
        movies = movies_data.get("movies", []) if isinstance(movies_data, dict) else movies_data
        documents = {
            str(movie["id"]): f"{movie.get('title', '')}\n{movie.get('description', '')}"
            for movie in movies
            if "id" in movie
        }
    except FileNotFoundError:
        documents = {
            "10": "The quick brown fox jumps over the lazy dog",
            "20": "Dogs are running and jumping in the park",
            "30": "Foxes love running fast"
        }
    semantic_search.load_or_create_embeddings(documents)
    return semantic_search.search(query, limit)

def verify_embeddings(save_dir: str = "cache"):
    import json
    import sys
    data_file = "data/movies.json"
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            movies_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Data file '{data_file}' is not valid JSON.", file=sys.stderr)
        sys.exit(1)
    
    movies = movies_data.get("movies", []) if isinstance(movies_data, dict) else movies_data
    documents = {
        str(movie["id"]): f"{movie.get('title', '')}\n{movie.get('description', '')}"
        for movie in movies
        if "id" in movie
    }

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(documents, save_dir)
    print(f"{semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


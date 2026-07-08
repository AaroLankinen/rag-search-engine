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
    
    def build_embeddings(self, documents, save_dir: str):
        self.documents = documents
        self.embeddings = self.model.encode(list(documents.values()), convert_to_numpy=True, show_progress_bar=True)
        self.document_map = {i: doc for i, doc in enumerate(documents.keys())}
        np.save(os.path.join(save_dir, "embeddings.npy"), self.embeddings)
        np.save(os.path.join(save_dir, "document_map.pkl"), self.document_map)
        return self.embeddings

    def load_or_create_embeddings(self, documents, save_dir: str):
        try:
            self.embeddings = np.load(os.path.join(save_dir, "embeddings.npy"))
            self.document_map = np.load(os.path.join(save_dir, "document_map.pkl"))
        except FileNotFoundError:
            self.build_embeddings(documents, save_dir)

    def generate_embedding(self, text):
        return self.model.encode(text, convert_to_numpy=True)

    def search(self, query: str, limit: int = 10) -> list[str]:
        query_embedding = self.generate_embedding(query)
        similarities = util.cos_sim(query_embedding, self.embeddings)[0]
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
    semantic_search.build_embeddings({
        "10": "The quick brown fox jumps over the lazy dog",
        "20": "Dogs are running and jumping in the park",
        "30": "Foxes love running fast"
    })
    return semantic_search.search(query, limit)

def verify_embeddings(save_dir: str = "cache"):
    import json
    data_file = "data/movies.json"
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            movies_data = json.load(f)
    except FileNotFoundError:
        movies_data = {"movies": []}
    
    movies = movies_data.get("movies", []) if isinstance(movies_data, dict) else movies_data
    documents = {
        str(movie["id"]): f"{movie.get('title', '')}\n{movie.get('description', '')}"
        for movie in movies
        if "id" in movie
    }

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(documents, save_dir)
    print(f"{semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions")
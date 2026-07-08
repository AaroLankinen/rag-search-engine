import os
import torch
from sentence_transformers import SentenceTransformer, util

class SemanticSearch:
    def __init__(self, documents: dict[str, str], model: str = "all-MiniLM-L6-v2"):
        self.documents = documents
        self.model = SentenceTransformer(model)
        self.document_embeddings = self.model.encode(documents.values(), convert_to_numpy=True)

def verify_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")
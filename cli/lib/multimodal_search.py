from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict

class MultimodalSearch:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path: str) -> np.ndarray:
        """Embed an image from a file path using the SentenceTransformer model."""
        try:
            image = Image.open(image_path)
            embedding = self.model.encode(image)
            if isinstance(embedding, np.ndarray):
                return embedding
            else:
                return np.array(embedding)
        except Exception as e:
            raise RuntimeError(f"Error loading or encoding image '{image_path}': {e}")

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text using the SentenceTransformer model."""
        embedding = self.model.encode(text)
        if isinstance(embedding, np.ndarray):
            return embedding
        else:
            return np.array(embedding)

def verify_image_embedding(image_path: str) -> None:
    embedder = MultimodalSearch()
    embedding = embedder.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search(image_path: str, movies: List[Dict[str, str]], k: int = 5) -> List[Dict]:
    searcher = MultimodalSearch()
    img_emb = searcher.embed_image(image_path)
    
    results = []
    for m in movies:
        text = f"{m.get('title', '')} {m.get('description', '')}"
        text_emb = searcher.embed_text(text)
        
        # Calculate cosine similarity
        norm_img = np.linalg.norm(img_emb)
        norm_txt = np.linalg.norm(text_emb)
        if norm_img > 0 and norm_txt > 0:
            sim = np.dot(img_emb, text_emb) / (norm_img * norm_txt)
        else:
            sim = 0.0
            
        results.append({
            "movie": m,
            "similarity": float(sim)
        })
        
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:k]
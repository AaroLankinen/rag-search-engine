from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np

class MultimodalSearch:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        return self.model.encode(image)

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text)

def verify_image_embedding(image_path: str) -> None:
    embedder = MultimodalSearch()
    embedding = embedder.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

if __name__ == "__main__":
    verify_image_embedding()
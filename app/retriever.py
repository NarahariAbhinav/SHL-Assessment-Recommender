import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class CatalogRetriever:
    def __init__(self, catalog_path=None, model_name="all-MiniLM-L6-v2"):
        if catalog_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            catalog_path = os.path.join(base_dir, "data", "shl_product_catalog_clean.json")
            
        self.model = SentenceTransformer(model_name)
        with open(catalog_path, 'r', encoding='utf-8') as f:
            self.catalog = json.load(f)
        
        self.documents = []
        self.doc_texts = []
        for item in self.catalog:
            # Build a rich text representation for the vector search
            name = item.get("name", "")
            desc = item.get("description", "")
            levels = item.get("job_levels_raw", "")
            keys = ", ".join(item.get("keys", []))
            
            text = f"Assessment Name: {name}\nDescription: {desc}\nTarget Job Levels: {levels}\nCategories/Keys: {keys}"
            self.doc_texts.append(text)
            self.documents.append(item)
            
        print("Building FAISS index...")
        self.embeddings = self.model.encode(self.doc_texts, show_progress_bar=False)
        self.dimension = self.embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(self.embeddings).astype('float32'))
        print(f"Index built with {self.index.ntotal} items.")

    def search(self, query: str, top_k: int = 10):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                item = self.documents[idx]
                results.append({
                    "name": item.get("name", ""),
                    "url": item.get("link", ""),
                    # Test type K=Knowledge, P=Personality, A=Aptitude etc. We can map from keys or default.
                    "test_type": self._extract_test_type(item),
                    "description": item.get("description", ""),
                    "score": float(distances[0][i])
                })
        return results

    def _extract_test_type(self, item) -> str:
        # The schema requires test_type. We map keys to characters based on typical SHL types if possible, or just default to "K"
        keys = " ".join(item.get("keys", [])).lower()
        if "personality" in keys or "behavior" in keys: return "P"
        if "ability" in keys or "aptitude" in keys: return "A"
        if "competencies" in keys: return "C"
        if "biodata" in keys: return "B"
        return "K" # Default Knowledge/Skill

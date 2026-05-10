import json
import os
import faiss
import numpy as np
import google.generativeai as genai

class CatalogRetriever:
    def __init__(self, catalog_path=None, model_name="models/gemini-embedding-2"):
        if catalog_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            catalog_path = os.path.join(base_dir, "data", "shl_product_catalog_clean.json")
            
        self.model_name = model_name
        
        # Configure genai in case it hasn't been globally yet
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)

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
            
        print("Building FAISS index with Gemini Embeddings...")
        
        # Batch embed 100 items at a time to stay under any payload limits
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(self.doc_texts), batch_size):
            batch = self.doc_texts[i:i+batch_size]
            response = genai.embed_content(model=self.model_name, content=batch)
            all_embeddings.extend(response['embedding'])
            
        self.embeddings = np.array(all_embeddings).astype('float32')
        self.dimension = self.embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings)
        print(f"Index built with {self.index.ntotal} items.")

    def search(self, query: str, top_k: int = 10):
        response = genai.embed_content(model=self.model_name, content=query)
        query_embedding = np.array([response['embedding']]).astype('float32')
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                item = self.documents[idx]
                results.append({
                    "name": item.get("name", ""),
                    "url": item.get("link", ""),
                    "test_type": self._extract_test_type(item),
                    "description": item.get("description", ""),
                    "score": float(distances[0][i])
                })
        return results

    def _extract_test_type(self, item) -> str:
        keys = " ".join(item.get("keys", [])).lower()
        if "personality" in keys or "behavior" in keys: return "P"
        if "ability" in keys or "aptitude" in keys: return "A"
        if "competencies" in keys: return "C"
        if "biodata" in keys: return "B"
        return "K"

import json
import os
import time
import numpy as np
import google.generativeai as genai

class CatalogRetriever:
    """
    Semantic search engine over the SHL product catalog.
    Uses Gemini Embeddings + NumPy cosine similarity (zero native dependencies).
    """
    def __init__(self, catalog_path=None, model_name="models/gemini-embedding-2"):
        if catalog_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            catalog_path = os.path.join(base_dir, "data", "shl_product_catalog_clean.json")
            
        self.model_name = model_name
        
        # Configure genai
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)

        with open(catalog_path, 'r', encoding='utf-8') as f:
            self.catalog = json.load(f)
        
        self.documents = []
        self.doc_texts = []
        for item in self.catalog:
            name = item.get("name", "")
            desc = item.get("description", "")
            levels = item.get("job_levels_raw", "")
            keys = ", ".join(item.get("keys", []))
            
            text = f"Assessment: {name}. {desc} Levels: {levels}. Tags: {keys}"
            self.doc_texts.append(text)
            self.documents.append(item)
        
        # Check for a pre-computed embedding cache to avoid re-embedding on every cold start
        cache_path = os.path.join(os.path.dirname(catalog_path), "embeddings_cache.npy")
        if os.path.exists(cache_path):
            print("Loading cached embeddings...")
            self.embeddings = np.load(cache_path)
        else:
            print("Building vector index with Gemini Embeddings...")
            self.embeddings = self._build_embeddings()
            # Save cache for future cold starts
            try:
                np.save(cache_path, self.embeddings)
                print("Saved embedding cache.")
            except Exception:
                pass  # read-only filesystem is fine, we'll just re-embed next time
        
        print(f"Index ready with {len(self.documents)} items ({self.embeddings.shape[1]}-dim).")

    def _build_embeddings(self):
        """Embed all catalog texts using Gemini, with rate-limit handling."""
        all_embeddings = []
        batch_size = 50  # smaller batches to stay under rate limits
        for i in range(0, len(self.doc_texts), batch_size):
            batch = self.doc_texts[i:i+batch_size]
            for attempt in range(5):
                try:
                    response = genai.embed_content(model=self.model_name, content=batch)
                    all_embeddings.extend(response['embedding'])
                    print(f"  Embedded {min(i+batch_size, len(self.doc_texts))}/{len(self.doc_texts)} items...")
                    break
                except Exception as e:
                    if "429" in str(e) or "ResourceExhausted" in str(e):
                        wait = 30 * (attempt + 1)
                        print(f"  Rate limited, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            # Small delay between batches to avoid hitting rate limits
            time.sleep(2)
            
        emb = np.array(all_embeddings, dtype=np.float32)
        # L2-normalize for cosine similarity via dot product
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / norms
        return emb

    def search(self, query: str, top_k: int = 10):
        """Retrieve the top-k most relevant catalog items for a query."""
        response = genai.embed_content(model=self.model_name, content=query)
        q = np.array(response['embedding'], dtype=np.float32)
        q = q / np.linalg.norm(q)
        
        # Cosine similarity = dot product of normalized vectors
        scores = self.embeddings @ q
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            item = self.documents[idx]
            results.append({
                "name": item.get("name", ""),
                "url": item.get("link", ""),
                "test_type": self._extract_test_type(item),
                "description": item.get("description", ""),
                "score": float(scores[idx])
            })
        return results

    def _extract_test_type(self, item) -> str:
        keys = " ".join(item.get("keys", [])).lower()
        if "personality" in keys or "behavior" in keys: return "P"
        if "ability" in keys or "aptitude" in keys: return "A"
        if "competencies" in keys: return "C"
        if "biodata" in keys: return "B"
        return "K"

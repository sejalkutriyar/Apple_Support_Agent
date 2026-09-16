import os
import pandas as pd
import numpy as np
import pickle

class HistoricalRetriever:
    """Retrieves top-k similar historically resolved AppleSupport threads for RAG grounding."""
    
    def __init__(self, 
                 data_path: str = "data/apple_support_sample.csv", 
                 cache_dir: str = "data/processed",
                 model_name: str = "all-MiniLM-L6-v2"):
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.embeddings_path = os.path.join(cache_dir, "resolved_embeddings.npy")
        self.metadata_path = os.path.join(cache_dir, "resolved_metadata.pkl")
        
        self.model = None
        self.use_tf_idf = False
        self.vectorizer = None
        
        self.df_resolved = None
        self.embeddings = None
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self._initialize_retriever()

    def _initialize_retriever(self):
        """Loads data, initializes embedding model, and builds/loads cached index."""
        print("Loading corpus dataset...")
        df = pd.read_csv(self.data_path)
        
        # Filter for publicly resolved threads (excluding pure DM-redirects for reply grounding)
        self.df_resolved = df[df['is_dm_redirect'] == False].copy().reset_index(drop=True)
        print(f"Total corpus size: {len(df)} | Publicly resolved grounding corpus size: {len(self.df_resolved)}")
        
        # Try loading sentence-transformers model
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading SentenceTransformer model '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"SentenceTransformer load failed ({e}). Falling back to TF-IDF retriever.")
            self.use_tf_idf = True
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

        # Build or load index
        if os.path.exists(self.embeddings_path) and os.path.exists(self.metadata_path):
            print("Loading precomputed vector embeddings from cache...")
            self.embeddings = np.load(self.embeddings_path)
            with open(self.metadata_path, "rb") as f:
                self.df_resolved = pickle.load(f)
        else:
            print("Precomputed index not found. Building fallback TF-IDF vector index for cloud environment...")
            self.use_tf_idf = True
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
            self._build_and_cache_index()

    def _build_and_cache_index(self):
        """Encodes customer messages in corpus and saves to local cache."""
        print("Computing embeddings for historical resolved cases...")
        texts = self.df_resolved['customer_message'].fillna("").tolist()
        
        if not self.use_tf_idf and self.model is not None:
            self.embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        else:
            self.embeddings = self.vectorizer.fit_transform(texts).toarray()
            # L2 normalize for cosine similarity via dot product
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = self.embeddings / norms

        np.save(self.embeddings_path, self.embeddings)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.df_resolved, f)
        print(f"Saved {len(self.embeddings)} embeddings to {self.embeddings_path}")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Searches historical resolved corpus for top_k most similar cases."""
        if not query or len(query.strip()) == 0:
            return []

        # Encode query
        if not self.use_tf_idf and self.model is not None:
            query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        else:
            query_emb = self.vectorizer.transform([query]).toarray()[0]
            norm = np.linalg.norm(query_emb)
            if norm > 0:
                query_emb = query_emb / norm

        # Compute Cosine Similarity via dot product (since normalized)
        similarities = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.df_resolved.iloc[idx]
            results.append({
                "case_id": str(row['root_id']),
                "customer_message": str(row['customer_message']),
                "resolution": str(row['final_resolution']),
                "similarity_score": float(similarities[idx]),
                "num_turns": int(row['num_turns'])
            })
        return results

    def get_max_similarity(self, query: str) -> float:
        """Returns maximum similarity score against corpus (used as OOD escalation signal)."""
        results = self.search(query, top_k=1)
        if results:
            return results[0]['similarity_score']
        return 0.0

if __name__ == "__main__":
    print("Testing HistoricalRetriever...")
    retriever = HistoricalRetriever()
    
    test_query = "@AppleSupport my iPhone battery is draining so fast after iOS 11 update!"
    print(f"\nQuery: '{test_query}'")
    
    results = retriever.search(test_query, top_k=3)
    for i, res in enumerate(results, 1):
        print(f"\n--- Match #{i} (Score: {res['similarity_score']:.4f}) ---")
        print(f"Case ID: {res['case_id']}")
        print(f"Customer: {res['customer_message']}")
        print(f"Historical Resolution: {res['resolution']}")

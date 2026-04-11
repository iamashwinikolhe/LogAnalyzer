"""Embedding and vector search functionality using FAISS."""

import numpy as np
from typing import List, Tuple, Optional
import faiss
from sentence_transformers import SentenceTransformer


class LogEmbedder:
    """Handles embedding generation and vector search for logs."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        
    def create_embeddings(self, documents: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of documents.
        
        Args:
            documents: List of text documents
            
        Returns:
            NumPy array of embeddings
        """
        embeddings = self.model.encode(documents, show_progress_bar=True)
        return np.array(embeddings).astype('float32')
    
    def build_index(self, documents: List[str]) -> None:
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of log chunks
        """
        self.documents = documents
        embeddings = self.create_embeddings(documents)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        print(f"✓ Built FAISS index with {len(documents)} documents")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.documents):
                # Convert L2 distance to similarity score (0 to 1)
                similarity = 1 / (1 + distance)
                results.append((self.documents[idx], similarity))
        
        return results
    
    def save_index(self, index_path: str) -> None:
        """
        Save the FAISS index to disk.
        
        Args:
            index_path: Path to save the index
        """
        if self.index is None:
            raise RuntimeError("No index to save. Build an index first.")
        
        faiss.write_index(self.index, index_path)
        print(f"✓ Saved index to {index_path}")
    
    def load_index(self, index_path: str, documents_list: List[str]) -> None:
        """
        Load a previously saved FAISS index.
        
        Args:
            index_path: Path to the saved index
            documents_list: List of documents corresponding to the index
        """
        self.index = faiss.read_index(index_path)
        self.documents = documents_list
        print(f"✓ Loaded index from {index_path}")

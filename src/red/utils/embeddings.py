import hashlib
import logging
import os
import pickle

from typing import List, Dict, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..config.config_loader import get_config

logger = logging.getLogger(__name__)

class EmbeddingProvider:
    """
    Unified interface for generating and managing text embeddings.
    
    Supports caching, semantic search, and multiple embedding models.
    """
    
    def __init__(self, 
                 model_name: str = None,
                 cache_dir: str = None,
                 enable_cache: bool = None,
                 device: str = None):
        """
        Initialize the embedding provider.
        
        Args:
            model_name: Name of the sentence transformer model (from config if None)
            cache_dir: Directory to store embedding cache (from config if None)
            enable_cache: Whether to enable embedding caching (from config if None)
            device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        # Load configuration
        config = get_config()
        embeddings_config = config.get('embeddings', {})
        data_config = config.get('data', {})
        
        # Set parameters from config or provided values
        self.model_name = model_name or embeddings_config.get('model_name', 'all-MiniLM-L6-v2')
        self.cache_dir = cache_dir or os.path.join(
            data_config.get('data_dir', './Datasets'), 
            embeddings_config.get('cache_dir', 'embeddings_cache')
        )
        self.enable_cache = enable_cache if enable_cache is not None else embeddings_config.get('enable_cache', True)
        
        # Initialize the embedding model with optimizations for Qwen
        if "Qwen" in self.model_name:
            # Optimize Qwen models with flash attention and left padding
            try:
                self.model = SentenceTransformer(
                    self.model_name,
                    model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
                    tokenizer_kwargs={"padding_side": "left"},
                    device=device
                )
                logger.info(f"Loaded Qwen model {self.model_name} with flash attention optimizations")
            except Exception as e:
                logger.warning(f"Failed to load Qwen model with optimizations, falling back to standard loading: {e}")
                self.model = SentenceTransformer(self.model_name, device=device)
        else:
            self.model = SentenceTransformer(self.model_name, device=device)
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Cache for embeddings
        self.embedding_cache = {}
        # Fix anti-pattern: use self.cache_dir and self.model_name instead of parameters
        self.cache_file = os.path.join(self.cache_dir, f"{self.model_name.replace('/', '_')}_cache.pkl")
        
        # FAISS index for fast similarity search
        self.faiss_index = None
        self.indexed_texts = []
        self.indexed_embeddings = {}
        
        # Setup cache directory and load existing cache
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._load_cache()
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text string."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Could not load embedding cache: {e}")
                self.embedding_cache = {}
    
    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        if self.enable_cache and self.embedding_cache:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.embedding_cache, f)
            except Exception as e:
                logger.warning(f"Could not save embedding cache: {e}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        cache_key = self._get_cache_key(text)
        
        # Check cache first
        if self.enable_cache and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Generate embedding with Qwen-specific optimizations
        if "Qwen" in self.model_name:
            # Use document prompt for Qwen models and enable truncation
            config = get_config()
            embeddings_config = config.get('embeddings', {})
            truncate_dim = embeddings_config.get('qwen_truncate_dim', 128)
            embedding = self.model.encode([text], prompt_name="document", truncate_dim=truncate_dim)[0]
        else:
            embedding = self.model.encode([text])[0]
        
        # Cache the embedding
        if self.enable_cache:
            self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def get_embeddings(self, texts: List[str], show_progress: bool = True) -> List[np.ndarray]:
        """
        Get embeddings for multiple text strings.
        
        Args:
            texts: List of input text strings
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Check which texts need new embeddings
        cache_keys = [self._get_cache_key(text) for text in texts]
        cached_embeddings = {}
        texts_to_encode = []
        indices_to_encode = []
        
        for i, (text, cache_key) in enumerate(zip(texts, cache_keys)):
            if self.enable_cache and cache_key in self.embedding_cache:
                cached_embeddings[i] = self.embedding_cache[cache_key]
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        # Generate embeddings for uncached texts
        new_embeddings = {}
        if texts_to_encode:
            logger.info(f"Generating embeddings for {len(texts_to_encode)} new texts...")
            # Generate embeddings with Qwen-specific optimizations
            if "Qwen" in self.model_name:
                # Use document prompt for Qwen models and enable truncation
                config = get_config()
                embeddings_config = config.get('embeddings', {})
                truncate_dim = embeddings_config.get('qwen_truncate_dim', 128)
                encoded = self.model.encode(texts_to_encode, prompt_name="document", truncate_dim=truncate_dim, show_progress_bar=show_progress)
            else:
                encoded = self.model.encode(texts_to_encode, show_progress_bar=show_progress)
            
            for i, (text, embedding) in enumerate(zip(texts_to_encode, encoded)):
                idx = indices_to_encode[i]
                new_embeddings[idx] = embedding
                
                # Cache the new embedding
                if self.enable_cache:
                    cache_key = cache_keys[idx]
                    self.embedding_cache[cache_key] = embedding
        
        # Combine cached and new embeddings in original order
        all_embeddings = []
        for i in range(len(texts)):
            if i in cached_embeddings:
                all_embeddings.append(cached_embeddings[i])
            else:
                all_embeddings.append(new_embeddings[i])
        
        # Save cache if new embeddings were generated
        if new_embeddings and self.enable_cache:
            self._save_cache()
        
        return all_embeddings
    
    def build_index(self, texts: List[str], embeddings: List[np.ndarray] = None) -> None:
        """
        Build a FAISS index for fast similarity search.
        
        Args:
            texts: List of texts to index
            embeddings: Pre-computed embeddings (optional)
        """
        if not texts:
            return
        
        print(f"Building FAISS index for {len(texts)} texts...")
        
        # Get embeddings if not provided
        if embeddings is None:
            embeddings = self.get_embeddings(texts)
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embedding_matrix)
        self.faiss_index.add(embedding_matrix)
        
        # Store texts and embeddings for retrieval
        self.indexed_texts = texts.copy()
        self.indexed_embeddings = {text: emb for text, emb in zip(texts, embeddings)}
        
        print(f"FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def find_similar(self, 
                    query_text: str, 
                    k: int = 5, 
                    threshold: float = 0.0) -> List[Dict[str, Union[str, float]]]:
        """
        Find similar texts using FAISS index.
        
        Args:
            query_text: Text to find similarities for
            k: Number of similar texts to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of dictionaries with 'text' and 'similarity' keys
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_index() first.")
        
        # Get query embedding
        query_embedding = self.get_embedding(query_text).astype('float32').reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.faiss_index.search(query_embedding, k)
        
        # Format results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= threshold and idx < len(self.indexed_texts):
                results.append({
                    'text': self.indexed_texts[idx],
                    'similarity': float(sim)
                })
        
        return results
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score
        """
        emb1 = self.get_embedding(text1).reshape(1, -1)
        emb2 = self.get_embedding(text2).reshape(1, -1)
        
        return cosine_similarity(emb1, emb2)[0][0]
    
    def find_similar_simple(self, 
                          query_text: str, 
                          candidate_texts: List[str], 
                          k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        Find similar texts without FAISS index (for small datasets).
        
        Args:
            query_text: Text to find similarities for
            candidate_texts: List of candidate texts
            k: Number of similar texts to return
            
        Returns:
            List of dictionaries with 'text' and 'similarity' keys
        """
        if not candidate_texts:
            return []
        
        # Get embeddings
        query_embedding = self.get_embedding(query_text)
        candidate_embeddings = self.get_embeddings(candidate_texts)
        
        # Compute similarities
        similarities = []
        for text, embedding in zip(candidate_texts, candidate_embeddings):
            sim = cosine_similarity(
                query_embedding.reshape(1, -1), 
                embedding.reshape(1, -1)
            )[0][0]
            similarities.append({'text': text, 'similarity': sim})
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:k]
    
    def save_index(self, filepath: str) -> None:
        """
        Save the FAISS index and associated data to disk.
        
        Args:
            filepath: Path to save the index
        """
        if self.faiss_index is None:
            raise ValueError("No index to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, f"{filepath}.faiss")
        
        # Save associated data
        data = {
            'indexed_texts': self.indexed_texts,
            'indexed_embeddings': self.indexed_embeddings,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim
        }
        
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str) -> None:
        """
        Load a FAISS index and associated data from disk.
        
        Args:
            filepath: Path to load the index from
        """
        # Load FAISS index
        self.faiss_index = faiss.read_index(f"{filepath}.faiss")
        
        # Load associated data
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        self.indexed_texts = data['indexed_texts']
        self.indexed_embeddings = data['indexed_embeddings']
        
        # Verify model compatibility
        if data['model_name'] != self.model_name:
            print(f"Warning: Loaded index was created with {data['model_name']}, "
                  f"but current model is {self.model_name}")
        
        print(f"Index loaded from {filepath} with {len(self.indexed_texts)} texts")
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embedding_cache = {}
        if self.enable_cache and os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the embedding cache."""
        return {
            'cached_embeddings': len(self.embedding_cache),
            'cache_enabled': self.enable_cache,
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim
        }
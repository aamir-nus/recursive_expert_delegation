"""
Label Subsetter for Greedy Subset Selection

This module implements the greedy subset selection algorithm that partitions
a large set of class labels into smaller subsets where each subset contains
labels that are maximally dissimilar to each other.
"""

import os
import pickle
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import umap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..config.config_loader import get_config

class LabelSubsetter:
    """
    Implements greedy subset selection for partitioning class labels.
    
    This class takes a large number of class labels, generates embeddings for them,
    optionally applies dimensionality reduction, and then creates subsets where
    each subset contains maximally dissimilar labels.
    """
    
    def __init__(self, 
                 embedding_model: str = None,
                 use_umap: bool = None,
                 umap_components: int = None,
                 random_state: int = None):
        """
        Initialize the LabelSubsetter.
        
        Args:
            embedding_model: Name of the sentence transformer model to use (from config if None)
            use_umap: Whether to apply UMAP dimensionality reduction (from config if None)
            umap_components: Number of UMAP components to use (from config if None)
            random_state: Random state for reproducibility (from config if None)
        """
        # Load configuration
        config = get_config()
        embeddings_config = config.get('embeddings', {})
        subsetting_config = config.get('subsetting', {})
        
        self.embedding_model_name = embedding_model or embeddings_config.get('model_name', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.use_umap = use_umap if use_umap is not None else subsetting_config.get('use_umap', True)
        self.umap_components = umap_components or subsetting_config.get('umap_components', 50)
        self.random_state = random_state or subsetting_config.get('random_state', 42)
        
        if self.use_umap:
            self.umap_reducer = umap.UMAP(
                n_components=self.umap_components,
                random_state=self.random_state,
                metric='cosine'
            )
        else:
            self.umap_reducer = None
            
        self.label_embeddings = {}
        self.subsets = []
        self.subset_mapping = {}  # Maps subset_id to list of labels
        
    def _generate_embeddings(self, labels: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a list of labels.
        
        Args:
            labels: List of class label strings
            
        Returns:
            Dictionary mapping labels to their embeddings
        """
        print(f"Generating embeddings for {len(labels)} labels...")
        embeddings = self.embedding_model.encode(labels, show_progress_bar=True)
        
        if self.use_umap and len(labels) > self.umap_components:
            print("Applying UMAP dimensionality reduction...")
            embeddings = self.umap_reducer.fit_transform(embeddings)
            
        return {label: embedding for label, embedding in zip(labels, embeddings)}
    
    def _avg_embedding(self, candidate_embeddings: List[np.ndarray]) -> np.ndarray:
        """Calculate the average embedding of a list of embeddings."""
        return np.mean(candidate_embeddings, axis=0)
    
    def _get_least_similar_embedding(self, 
                                   target_embedding: np.ndarray, 
                                   candidate_embeddings: List[np.ndarray]) -> Tuple[int, np.ndarray]:
        """
        Find the embedding that is least similar to the target embedding.
        
        Args:
            target_embedding: The reference embedding
            candidate_embeddings: List of candidate embeddings to compare against
            
        Returns:
            Tuple of (index, embedding) of the least similar candidate
        """
        if len(candidate_embeddings) == 0:
            raise ValueError("No candidate embeddings provided")
            
        # Reshape target embedding for cosine similarity calculation
        target_embedding = target_embedding.reshape(1, -1)
        candidate_array = np.array(candidate_embeddings)
        
        similarities = cosine_similarity(target_embedding, candidate_array)[0]
        least_similar_index = np.argmin(similarities)
        
        return least_similar_index, candidate_embeddings[least_similar_index]
    
    def _get_embedding_label(self, 
                           embedding: np.ndarray, 
                           embedding_map: Dict[str, np.ndarray]) -> Optional[str]:
        """
        Find the label corresponding to a given embedding.
        
        Args:
            embedding: The embedding to find the label for
            embedding_map: Dictionary mapping labels to embeddings
            
        Returns:
            The corresponding label, or None if not found
        """
        for label, emb in embedding_map.items():
            if np.array_equal(embedding, emb):
                return label
        return None
    
    def create_subsets(self, 
                      labels: List[str], 
                      subset_size: int = 8) -> Dict[str, List[str]]:
        """
        Create subsets of labels using the greedy subset selection algorithm.
        
        Args:
            labels: List of class label strings
            subset_size: Maximum number of labels per subset
            
        Returns:
            Dictionary mapping subset IDs to lists of labels
        """
        print(f"Creating subsets from {len(labels)} labels with max size {subset_size}")
        
        # Generate embeddings for all labels
        self.label_embeddings = self._generate_embeddings(labels)
        
        # Track which labels have been assigned to subsets
        visited = {label: False for label in labels}
        subsets = []
        current_subset = []
        
        while any(not visited[label] for label in visited):
            # Find an unvisited label to start or continue the current subset
            available_labels = [label for label in labels if not visited[label]]
            if not available_labels:
                break
                
            if not current_subset:
                # Start new subset with first available label
                first_label = available_labels[0]
                current_subset.append(first_label)
                visited[first_label] = True
                
            elif len(current_subset) >= subset_size:
                # Current subset is full, start a new one
                subsets.append(current_subset.copy())
                current_subset = []
                
            else:
                # Add the most dissimilar label to current subset
                current_subset_embeddings = [self.label_embeddings[label] for label in current_subset]
                subset_avg = self._avg_embedding(current_subset_embeddings)
                
                # Get embeddings of remaining unvisited labels
                remaining_embeddings = [self.label_embeddings[label] for label in available_labels]
                
                if not remaining_embeddings:
                    break
                
                # Find the least similar embedding
                least_similar_idx, least_similar_emb = self._get_least_similar_embedding(
                    subset_avg, remaining_embeddings
                )
                
                # Find the corresponding label
                corresponding_label = available_labels[least_similar_idx]
                
                # Add to current subset and mark as visited
                current_subset.append(corresponding_label)
                visited[corresponding_label] = True
        
        # Add any remaining labels in the current subset
        if current_subset:
            subsets.append(current_subset)
        
        # Create subset mapping with IDs
        self.subset_mapping = {f"subset_{i}": subset for i, subset in enumerate(subsets)}
        self.subsets = subsets
        
        print(f"Created {len(self.subsets)} subsets:")
        for subset_id, labels_in_subset in self.subset_mapping.items():
            print(f"  {subset_id}: {len(labels_in_subset)} labels")
            
        return self.subset_mapping
    
    def get_subset_for_label(self, label: str) -> Optional[str]:
        """
        Find which subset a given label belongs to.
        
        Args:
            label: The label to find
            
        Returns:
            The subset ID containing the label, or None if not found
        """
        for subset_id, labels_in_subset in self.subset_mapping.items():
            if label in labels_in_subset:
                return subset_id
        return None
    
    def save(self, filepath: str) -> None:
        """
        Save the subsetter state to disk.
        
        Args:
            filepath: Path to save the subsetter state
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        state = {
            'embedding_model_name': self.embedding_model_name,
            'use_umap': self.use_umap,
            'umap_components': self.umap_components,
            'random_state': self.random_state,
            'label_embeddings': self.label_embeddings,
            'subsets': self.subsets,
            'subset_mapping': self.subset_mapping,
            'umap_reducer': self.umap_reducer
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Subsetter saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LabelSubsetter':
        """
        Load a subsetter from disk.
        
        Args:
            filepath: Path to load the subsetter from
            
        Returns:
            Loaded LabelSubsetter instance
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Create new instance
        subsetter = cls(
            embedding_model=state['embedding_model_name'],
            use_umap=state['use_umap'],
            umap_components=state['umap_components'],
            random_state=state['random_state']
        )
        
        # Restore state
        subsetter.label_embeddings = state['label_embeddings']
        subsetter.subsets = state['subsets']
        subsetter.subset_mapping = state['subset_mapping']
        subsetter.umap_reducer = state['umap_reducer']
        
        print(f"Subsetter loaded from {filepath}")
        return subsetter
    
    def get_subset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the created subsets.
        
        Returns:
            Dictionary containing subset statistics
        """
        if not self.subsets:
            return {"error": "No subsets created yet"}
        
        subset_sizes = [len(subset) for subset in self.subsets]
        
        return {
            "num_subsets": len(self.subsets),
            "total_labels": sum(subset_sizes),
            "avg_subset_size": np.mean(subset_sizes),
            "min_subset_size": min(subset_sizes),
            "max_subset_size": max(subset_sizes),
            "subset_sizes": subset_sizes
        }

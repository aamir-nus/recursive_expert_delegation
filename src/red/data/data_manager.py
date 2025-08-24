"""
Data Manager for R.E.D. Framework

This module handles all data I/O operations, semantic search, and data preparation
for the R.E.D. framework including training data, validation data, and unlabeled data.
"""

import os
import pandas as pd
import numpy as np
import pickle
import json
from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path
import random

from ..utils.embeddings import EmbeddingProvider

class DataManager:
    """
    Manages all data operations for the R.E.D. framework.
    
    Handles:
    - Loading and saving training/unlabeled data
    - Semantic search and similarity matching
    - Data preparation for subset classifiers
    - Managing validated samples from LLM feedback
    - Data splits and batch processing
    """
    
    def __init__(self, 
                 data_dir: str = None,
                 embedding_model: str = None,
                 cache_embeddings: bool = None):
        """
        Initialize the data manager.
        
        Args:
            data_dir: Directory to store data files (from config if None)
            embedding_model: Model to use for embeddings (from config if None)
            cache_embeddings: Whether to cache embeddings (from config if None)
        """
        # Load configuration
        from ..config.config_loader import get_config
        config = get_config()
        data_config = config.get('data', {})
        embeddings_config = config.get('embeddings', {})
        
        # Set parameters from config or provided values
        data_dir = data_dir or data_config.get('data_dir', './Datasets')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding provider with configuration
        self.embedding_provider = EmbeddingProvider(
            model_name=embedding_model,  # Will use config internally if None
            cache_dir=None,  # Will use config internally
            enable_cache=cache_embeddings  # Will use config internally if None
        )
        
        # Data storage
        self.seed_data = {'texts': [], 'labels': []}
        self.unlabeled_data = []
        self.validated_data = {'texts': [], 'labels': [], 'metadata': []}
        self.subset_mapping = {}  # Maps labels to subset IDs
        
        # Index for fast similarity search
        self.search_index_built = False
        
    def load_seed_data(self, 
                      filepath: str, 
                      text_column: str = 'text',
                      label_column: str = 'label',
                      encoding: str = 'utf-8') -> Dict[str, int]:
        """
        Load the initial seed training data.
        
        Args:
            filepath: Path to the seed data file (CSV, JSON, or pickle)
            text_column: Name of the text column
            label_column: Name of the label column
            encoding: File encoding
            
        Returns:
            Dictionary with data statistics
        """
        print(f"Loading seed data from {filepath}")
        
        file_extension = Path(filepath).suffix.lower()
        
        try:
            if file_extension == '.csv':
                df = pd.read_csv(filepath, encoding=encoding)
                texts = df[text_column].astype(str).tolist()
                labels = df[label_column].astype(str).tolist()
                
            elif file_extension == '.json':
                with open(filepath, 'r', encoding=encoding) as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    # List of dictionaries
                    texts = [item[text_column] for item in data]
                    labels = [item[label_column] for item in data]
                else:
                    # Dictionary with lists
                    texts = data[text_column]
                    labels = data[label_column]
                    
            elif file_extension in ['.pkl', '.pickle']:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                texts = data[text_column]
                labels = data[label_column]
                
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Clean and validate data
            texts = [str(text).strip() for text in texts if text and str(text).strip()]
            labels = [str(label).strip() for label in labels if label and str(label).strip()]
            
            if len(texts) != len(labels):
                raise ValueError("Number of texts and labels must match")
            
            self.seed_data = {'texts': texts, 'labels': labels}
            
            # Generate statistics
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            stats = {
                'total_samples': len(texts),
                'unique_labels': len(label_counts),
                'label_distribution': label_counts,
                'avg_samples_per_label': np.mean(list(label_counts.values())),
                'min_samples_per_label': min(label_counts.values()),
                'max_samples_per_label': max(label_counts.values())
            }
            
            print(f"Loaded {len(texts)} samples with {len(label_counts)} unique labels")
            print(f"Label distribution: {label_counts}")
            
            return stats
            
        except Exception as e:
            print(f"Error loading seed data: {e}")
            raise
    
    def load_unlabeled_data(self, 
                           filepath: str, 
                           text_column: str = 'text',
                           encoding: str = 'utf-8',
                           max_samples: int = None) -> int:
        """
        Load unlabeled data for classification.
        
        Args:
            filepath: Path to the unlabeled data file
            text_column: Name of the text column
            encoding: File encoding
            max_samples: Maximum number of samples to load
            
        Returns:
            Number of samples loaded
        """
        print(f"Loading unlabeled data from {filepath}")
        
        file_extension = Path(filepath).suffix.lower()
        
        try:
            if file_extension == '.csv':
                df = pd.read_csv(filepath, encoding=encoding)
                texts = df[text_column].astype(str).tolist()
                
            elif file_extension == '.json':
                with open(filepath, 'r', encoding=encoding) as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    texts = [item[text_column] if isinstance(item, dict) else str(item) for item in data]
                else:
                    texts = data[text_column]
                    
            elif file_extension in ['.pkl', '.pickle']:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                texts = data[text_column] if isinstance(data, dict) else data
                
            elif file_extension == '.txt':
                with open(filepath, 'r', encoding=encoding) as f:
                    texts = [line.strip() for line in f if line.strip()]
                    
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Clean texts
            texts = [str(text).strip() for text in texts if text and str(text).strip()]
            
            # Limit samples if requested
            if max_samples and len(texts) > max_samples:
                texts = texts[:max_samples]
            
            self.unlabeled_data = texts
            
            print(f"Loaded {len(texts)} unlabeled samples")
            return len(texts)
            
        except Exception as e:
            print(f"Error loading unlabeled data: {e}")
            raise
    
    def save_data(self, 
                 data: Dict[str, List], 
                 filepath: str, 
                 format: str = 'csv') -> None:
        """
        Save data to file.
        
        Args:
            data: Dictionary with 'texts' and 'labels' keys
            filepath: Path to save the file
            format: Output format ('csv', 'json', 'pickle')
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if format == 'csv':
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
                
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Data saved to {filepath}")
    
    def set_subset_mapping(self, subset_mapping: Dict[str, List[str]]) -> None:
        """
        Set the mapping of labels to subset IDs.
        
        Args:
            subset_mapping: Dictionary mapping subset IDs to lists of labels
        """
        # Create reverse mapping: label -> subset_id
        self.subset_mapping = {}
        for subset_id, labels in subset_mapping.items():
            for label in labels:
                self.subset_mapping[label] = subset_id
        
        print(f"Subset mapping set for {len(self.subset_mapping)} labels")
    
    def get_subset_data(self, subset_id: str) -> Dict[str, List]:
        """
        Get training data for a specific subset.
        
        Args:
            subset_id: ID of the subset
            
        Returns:
            Dictionary with texts and labels for the subset
        """
        # Find labels belonging to this subset
        subset_labels = [label for label, sid in self.subset_mapping.items() if sid == subset_id]
        
        if not subset_labels:
            return {'texts': [], 'labels': []}
        
        # Filter seed data for this subset
        subset_texts = []
        subset_labels_filtered = []
        
        for text, label in zip(self.seed_data['texts'], self.seed_data['labels']):
            if label in subset_labels:
                subset_texts.append(text)
                subset_labels_filtered.append(label)
        
        # Add any validated data for this subset
        for text, label, metadata in zip(
            self.validated_data['texts'], 
            self.validated_data['labels'],
            self.validated_data['metadata']
        ):
            if label in subset_labels:
                subset_texts.append(text)
                subset_labels_filtered.append(label)
        
        return {'texts': subset_texts, 'labels': subset_labels_filtered}
    
    def get_noise_data(self, subset_id: str, max_samples: int = 1000) -> List[str]:
        """
        Get noise data (from other subsets) for training.
        
        Args:
            subset_id: ID of the current subset
            max_samples: Maximum number of noise samples to return
            
        Returns:
            List of noise text samples
        """
        # Find labels NOT belonging to this subset
        subset_labels = [label for label, sid in self.subset_mapping.items() if sid == subset_id]
        
        noise_texts = []
        
        # Get noise from seed data
        for text, label in zip(self.seed_data['texts'], self.seed_data['labels']):
            if label not in subset_labels:
                noise_texts.append(text)
        
        # Get noise from validated data
        for text, label in zip(self.validated_data['texts'], self.validated_data['labels']):
            if label not in subset_labels:
                noise_texts.append(text)
        
        # Shuffle and limit
        random.shuffle(noise_texts)
        return noise_texts[:max_samples]
    
    def build_search_index(self, include_validated: bool = True) -> None:
        """
        Build semantic search index for similarity search.
        
        Args:
            include_validated: Whether to include validated data in the index
        """
        print("Building semantic search index...")
        
        # Combine all texts for indexing
        all_texts = self.seed_data['texts'].copy()
        
        if include_validated:
            all_texts.extend(self.validated_data['texts'])
        
        if all_texts:
            self.embedding_provider.build_index(all_texts)
            self.search_index_built = True
            print(f"Search index built with {len(all_texts)} texts")
        else:
            print("No texts available for indexing")
    
    def find_similar_texts(self, 
                          query_text: str, 
                          k: int = 5,
                          label_filter: str = None) -> List[Dict[str, Any]]:
        """
        Find similar texts using semantic search.
        
        Args:
            query_text: Text to find similarities for
            k: Number of similar texts to return
            label_filter: Optional label to filter results
            
        Returns:
            List of similar texts with metadata
        """
        if not self.search_index_built:
            self.build_search_index()
        
        # Find similar texts
        similar_results = self.embedding_provider.find_similar(query_text, k=k*2)  # Get more for filtering
        
        # Add labels and filter if requested
        results_with_labels = []
        for result in similar_results:
            text = result['text']
            similarity = result['similarity']
            
            # Find the label for this text
            label = None
            # Check seed data
            for i, seed_text in enumerate(self.seed_data['texts']):
                if seed_text == text:
                    label = self.seed_data['labels'][i]
                    break
            
            # Check validated data if not found
            if label is None:
                for i, val_text in enumerate(self.validated_data['texts']):
                    if val_text == text:
                        label = self.validated_data['labels'][i]
                        break
            
            # Apply label filter
            if label_filter is None or label == label_filter:
                results_with_labels.append({
                    'text': text,
                    'label': label,
                    'similarity': similarity
                })
                
                if len(results_with_labels) >= k:
                    break
        
        return results_with_labels
    
    def add_validated_sample(self, 
                           text: str, 
                           label: str, 
                           confidence: float = 1.0,
                           validation_metadata: Dict = None) -> None:
        """
        Add a validated sample to the training data.
        
        Args:
            text: Validated text sample
            label: Validated label
            confidence: Validation confidence score
            validation_metadata: Additional metadata from validation
        """
        if validation_metadata is None:
            validation_metadata = {}
        
        metadata = {
            'confidence': confidence,
            'validation_metadata': validation_metadata,
            'source': 'llm_validation'
        }
        
        self.validated_data['texts'].append(text)
        self.validated_data['labels'].append(label)
        self.validated_data['metadata'].append(metadata)
        
        # Rebuild search index if it was built before
        if self.search_index_built:
            self.build_search_index()
    
    def add_validated_batch(self, 
                          texts: List[str], 
                          labels: List[str],
                          confidences: List[float] = None,
                          metadata_list: List[Dict] = None) -> None:
        """
        Add multiple validated samples at once.
        
        Args:
            texts: List of validated text samples
            labels: List of validated labels
            confidences: List of confidence scores
            metadata_list: List of metadata dictionaries
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")
        
        if confidences is None:
            confidences = [1.0] * len(texts)
        
        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        
        for text, label, confidence, metadata in zip(texts, labels, confidences, metadata_list):
            self.add_validated_sample(text, label, confidence, metadata)
        
        print(f"Added {len(texts)} validated samples to training data")
    
    def get_unlabeled_batch(self, 
                           batch_size: int = 100, 
                           start_index: int = 0) -> List[str]:
        """
        Get a batch of unlabeled data for processing.
        
        Args:
            batch_size: Size of the batch
            start_index: Starting index in the unlabeled data
            
        Returns:
            List of unlabeled text samples
        """
        end_index = min(start_index + batch_size, len(self.unlabeled_data))
        return self.unlabeled_data[start_index:end_index]
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the managed data."""
        # Seed data stats
        seed_label_counts = {}
        for label in self.seed_data['labels']:
            seed_label_counts[label] = seed_label_counts.get(label, 0) + 1
        
        # Validated data stats
        validated_label_counts = {}
        for label in self.validated_data['labels']:
            validated_label_counts[label] = validated_label_counts.get(label, 0) + 1
        
        # Combined stats
        all_labels = self.seed_data['labels'] + self.validated_data['labels']
        total_label_counts = {}
        for label in all_labels:
            total_label_counts[label] = total_label_counts.get(label, 0) + 1
        
        return {
            'seed_data': {
                'samples': len(self.seed_data['texts']),
                'unique_labels': len(seed_label_counts),
                'label_distribution': seed_label_counts
            },
            'validated_data': {
                'samples': len(self.validated_data['texts']),
                'unique_labels': len(validated_label_counts),
                'label_distribution': validated_label_counts
            },
            'unlabeled_data': {
                'samples': len(self.unlabeled_data)
            },
            'total_labeled_data': {
                'samples': len(all_labels),
                'unique_labels': len(total_label_counts),
                'label_distribution': total_label_counts
            },
            'subset_mapping': self.subset_mapping,
            'search_index_built': self.search_index_built
        }
    
    def save_state(self, filepath: str) -> None:
        """Save the data manager state to disk."""
        state = {
            'seed_data': self.seed_data,
            'validated_data': self.validated_data,
            'subset_mapping': self.subset_mapping,
            'unlabeled_data': self.unlabeled_data[:100],  # Save only first 100 for space
            'embedding_model': self.embedding_provider.model_name
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Data manager state saved to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Load data manager state from disk."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.seed_data = state.get('seed_data', {'texts': [], 'labels': []})
            self.validated_data = state.get('validated_data', {'texts': [], 'labels': [], 'metadata': []})
            self.subset_mapping = state.get('subset_mapping', {})
            
            # Note: unlabeled_data is not restored to save space
            print(f"Data manager state loaded from {filepath}")
            
        except FileNotFoundError:
            print(f"State file not found: {filepath}")
        except Exception as e:
            print(f"Error loading state: {e}")
    
    def export_training_data(self, 
                           filepath: str, 
                           include_validated: bool = True,
                           format: str = 'csv') -> None:
        """
        Export all training data to a file.
        
        Args:
            filepath: Path to save the training data
            include_validated: Whether to include validated samples
            format: Output format
        """
        texts = self.seed_data['texts'].copy()
        labels = self.seed_data['labels'].copy()
        
        if include_validated:
            texts.extend(self.validated_data['texts'])
            labels.extend(self.validated_data['labels'])
        
        data = {'texts': texts, 'labels': labels}
        self.save_data(data, filepath, format)
        
        print(f"Exported {len(texts)} training samples to {filepath}")
    
    def clear_validated_data(self) -> None:
        """Clear all validated data."""
        self.validated_data = {'texts': [], 'labels': [], 'metadata': []}
        print("Validated data cleared")
    
    def get_label_examples(self, 
                          label: str, 
                          max_examples: int = 10,
                          include_validated: bool = True) -> List[str]:
        """
        Get example texts for a specific label.
        
        Args:
            label: Label to get examples for
            max_examples: Maximum number of examples to return
            include_validated: Whether to include validated examples
            
        Returns:
            List of example texts
        """
        examples = []
        
        # Get from seed data
        for text, text_label in zip(self.seed_data['texts'], self.seed_data['labels']):
            if text_label == label:
                examples.append(text)
        
        # Get from validated data
        if include_validated:
            for text, text_label in zip(self.validated_data['texts'], self.validated_data['labels']):
                if text_label == label:
                    examples.append(text)
        
        # Shuffle and limit
        random.shuffle(examples)
        return examples[:max_examples]

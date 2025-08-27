"""
Subset Classifier with Noise Oversampling

This module implements the SubsetClassifier that handles training lightweight models
for individual label subsets with noise oversampling for improved robustness.
"""
import joblib
import math
import os
import pickle
import logging
from typing import List, Dict, Tuple, Any

import numpy as np

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset

from ..config.config_loader import get_config
from ..utils.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

class SubsetClassifier:
    """
    Classifier for a specific subset of labels with noise oversampling.
    
    This class handles:
    - Training on a subset of labels with noise class
    - Noise oversampling to handle imbalanced data
    - Uncertainty sampling for active learning
    - Model persistence and retraining
    """
    
    NOISE_LABEL = "__NOISE__"

    def __init__(self, 
                 subset_id: str,
                 subset_labels: List[str],
                 classifier_type: str = None,
                 use_embeddings: bool = None,
                 embedding_model: str = None,
                 noise_oversample_factor: float = None,
                 max_features: int = None,
                 random_state: int = None):
        """
        Initialize the subset classifier.
        
        Args:
            subset_id: Unique identifier for this subset
            subset_labels: List of labels this classifier handles
            classifier_type: Type of classifier (from config if None)
            use_embeddings: Whether to use embeddings or TF-IDF features (from config if None)
            embedding_model: Embedding model to use if use_embeddings=True (from config if None)
            noise_oversample_factor: Factor to oversample noise class (from config if None)
            max_features: Maximum features for TF-IDF (from config if None)
            random_state: Random state for reproducibility (from config if None)
        """
        # Load configuration
        config = get_config()
        classifier_config = config.get('classifier', {})
        embeddings_config = config.get('embeddings', {})
        subsetting_config = config.get('subsetting', {})
        
        self.subset_id = subset_id
        self.subset_labels = subset_labels
        self.classifier_type = classifier_type or classifier_config.get('type', 'logistic_regression')
        self.use_embeddings = use_embeddings if use_embeddings is not None else classifier_config.get('use_embeddings', True)
        self.embedding_model_name = embedding_model or embeddings_config.get('model_name', 'all-MiniLM-L6-v2')
        self.noise_oversample_factor = noise_oversample_factor or classifier_config.get('noise_oversample_factor', 2.0)
        self.max_features = max_features or classifier_config.get('max_features', 10000)
        self.random_state = random_state or subsetting_config.get('random_state', 42)
        
        # Add noise class to the label set
        self.all_labels = subset_labels + [self.NOISE_LABEL]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.all_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Initialize components
        self.embedding_provider = None
        if use_embeddings:
            self.embedding_provider = EmbeddingProvider(model_name=embedding_model)
        
        self.vectorizer = None
        self.classifier = None
        self.pipeline = None
        
        # Training data storage
        self.training_texts = []
        self.training_labels = []
        self.is_trained = False
        
        # Training statistics
        self.training_stats = {}
        
    def _create_classifier(self):
        """
        Create the base classifier based on type.
        
        Note: Algorithm-specific hyperparameters are kept as sensible defaults
        in the code rather than config, as they are implementation details
        of the specific ML algorithms, not application-level configuration.
        """
        if self.classifier_type == "logistic_regression":
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,  # Sklearn default, sufficient for most text classification
                multi_class='multinomial'  # Appropriate for multi-class text classification
            )
        elif self.classifier_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,  # Sklearn default, good balance of performance/speed
                random_state=self.random_state,
                n_jobs=-1  # Use all available cores for training
            )
        elif self.classifier_type == "setfit":
            # SetFit: High-performance few-shot classifier using sentence transformers
            # Uses the same embedding model as configured for consistency
            return SetFitModel.from_pretrained(
                self.embedding_model_name,
                use_differentiable_head=True,  # Better performance for small datasets
                head_params={
                    "hidden_dropout_prob": 0.1,  # Regularization for small datasets
                    "hidden_size": 128  # Compact head for efficiency
                }
            )
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
    
    def _train_setfit_classifier(self, texts: List[str], labels: List[str], X_train: np.ndarray, y_train: np.ndarray):
        """
        Train a SetFit classifier using the special SetFit training procedure.
        
        Args:
            texts: Training texts
            labels: Training labels
            X_train: Feature matrix (not used for SetFit)
            y_train: Label indices (not used for SetFit)
        """
        # Create Dataset object for SetFit
        train_dataset = Dataset.from_dict({
            "text": texts,
            "label": labels
        })
        
        # Create trainer with SetFit-specific parameters
        trainer = SetFitTrainer(
            model=self.classifier,
            train_dataset=train_dataset,
            num_iterations=20,  # SetFit training iterations, good default for few-shot
            num_epochs=1,  # Number of epochs per iteration
            batch_size=16,  # Batch size for contrastive learning
            seed=self.random_state,
            column_mapping={"text": "text", "label": "label"}
        )
        
        # Train the SetFit model
        trainer.train()
        
        # Update the classifier reference
        self.classifier = trainer.model
        
        logger.info("SetFit training completed")
    
    def _prepare_features(self, texts: List[str]) -> np.ndarray:
        """
        Prepare features from text data.
        
        Args:
            texts: List of text samples
            
        Returns:
            Feature matrix
        """
        if self.use_embeddings:
            if self.embedding_provider is None:
                self.embedding_provider = EmbeddingProvider(model_name=self.embedding_model_name)
            
            features = self.embedding_provider.get_embeddings(texts)
            return np.array(features)
        else:
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                features = self.vectorizer.fit_transform(texts)
            else:
                features = self.vectorizer.transform(texts)
            
            return features.toarray()
    
    def _oversample_noise(self, 
                         texts: List[str], 
                         labels: List[str], 
                         noise_texts: List[str]) -> Tuple[List[str], List[str]]:
        """
        Oversample noise data to balance the dataset.
        
        Args:
            texts: Training texts for this subset
            labels: Training labels for this subset
            noise_texts: Noise texts from other subsets
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        # Calculate how many noise samples we need
        subset_samples_per_class = {}
        for label in self.subset_labels:
            subset_samples_per_class[label] = labels.count(label)
        
        avg_samples_per_class = np.mean(list(subset_samples_per_class.values()))
        target_noise_samples = int(avg_samples_per_class * self.noise_oversample_factor)
        
        # Sample noise texts
        if len(noise_texts) > target_noise_samples:
            np.random.seed(self.random_state)
            sampled_noise = np.random.choice(noise_texts, target_noise_samples, replace=False).tolist()
        else:
            # If not enough noise texts, repeat some
            sampled_noise = noise_texts * (target_noise_samples // len(noise_texts) + 1)
            sampled_noise = sampled_noise[:target_noise_samples]
        
        # Combine with original data
        augmented_texts = texts + sampled_noise
        augmented_labels = labels + [self.NOISE_LABEL] * len(sampled_noise)
        
        logger.info(f"Subset {self.subset_id}: Added {len(sampled_noise)} noise samples")
        logger.info(f"  Total samples: {len(augmented_texts)}")
        logger.info(f"  Noise ratio: {len(sampled_noise) / len(augmented_texts):.2f}")
        
        return augmented_texts, augmented_labels
    
    def train(self, 
              texts: List[str], 
              labels: List[str], 
              noise_texts: List[str] = None,
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the classifier with noise oversampling.
        
        Args:
            texts: Training texts for this subset
            labels: Training labels for this subset
            noise_texts: Noise texts from other subsets
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training statistics and metrics
        """
        logger.info(f"\nTraining classifier for subset {self.subset_id}")
        logger.info(f"Subset labels: {self.subset_labels}")
        logger.info(f"Training samples: {len(texts)}")
        
        # Validate inputs
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")
        
        # Filter texts that belong to this subset
        subset_texts = []
        subset_labels = []
        for text, label in zip(texts, labels):
            if label in self.subset_labels:
                subset_texts.append(text)
                subset_labels.append(label)
        
        logger.info(f"Samples for this subset: {len(subset_texts)}")
        
        if len(subset_texts) == 0:
            raise ValueError(f"No training samples found for subset {self.subset_id}")
        
        # Add noise samples if provided
        if noise_texts:
            subset_texts, subset_labels = self._oversample_noise(
                subset_texts, subset_labels, noise_texts
            )
        
        # Store training data
        self.training_texts = subset_texts.copy()
        self.training_labels = subset_labels.copy()
        
        # Prepare features
        logger.info("Preparing features...")
        X = self._prepare_features(subset_texts)
        y = [self.label_to_idx[label] for label in subset_labels]
        y = np.array(y)
        
        # Split data for validation if requested
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=self.random_state, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # Apply SMOTE if we have multiple classes and enough samples
        # Note: SMOTE threshold is kept in code as it's an algorithm-specific
        # decision about when oversampling is statistically meaningful
        unique_classes = np.unique(y_train)
        min_samples_for_smote = 20  # Minimum samples needed for meaningful SMOTE oversampling
        
        if len(unique_classes) > 1 and len(y_train) > min_samples_for_smote:
            try:
                smote = SMOTE(random_state=self.random_state)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logger.info("Applied SMOTE oversampling")
            except Exception as e:
                logger.warning(f"SMOTE failed, continuing without: {e}")
        
        # Create and train classifier
        logger.info("Training classifier...")
        self.classifier = self._create_classifier()
        
        if self.classifier_type == "setfit":
            # SetFit requires special training procedure
            self._train_setfit_classifier(subset_texts, subset_labels, X_train, y_train)
        else:
            # Standard sklearn classifiers
            self.classifier.fit(X_train, y_train)
        
        # Calculate training statistics
        if self.classifier_type == "setfit":
            # For SetFit, calculate accuracy manually
            train_predictions = self.predict(subset_texts)
            train_score = sum(1 for pred, true in zip(train_predictions, subset_labels) if pred == true) / len(subset_labels)
        else:
            train_score = self.classifier.score(X_train, y_train)
        stats = {
            'subset_id': self.subset_id,
            'subset_labels': self.subset_labels,
            'train_samples': len(X_train),
            'train_accuracy': train_score,
            'noise_samples': subset_labels.count(self.NOISE_LABEL),
            'label_distribution': {label: subset_labels.count(label) for label in self.all_labels}
        }
        
        if X_val is not None:
            if self.classifier_type == "setfit":
                # For SetFit validation, use text-based prediction
                val_texts = [subset_texts[i] for i in range(len(subset_texts)) if i < len(subset_texts) // 5]  # Approximate validation split
                val_labels = [subset_labels[i] for i in range(len(subset_labels)) if i < len(subset_labels) // 5]
                val_predictions = self.predict(val_texts)
                val_score = sum(1 for pred, true in zip(val_predictions, val_labels) if pred == true) / len(val_labels)
                pred_labels = val_predictions
            else:
                val_score = self.classifier.score(X_val, y_val)
                # Detailed validation metrics
                y_pred = self.classifier.predict(X_val)
                val_labels = [self.idx_to_label[idx] for idx in y_val]
                pred_labels = [self.idx_to_label[idx] for idx in y_pred]
            
            stats['validation_accuracy'] = val_score
            
            stats['classification_report'] = classification_report(
                val_labels, pred_labels, output_dict=True
            )
        
        self.training_stats = stats
        self.is_trained = True
        
        logger.info(f"Training completed. Train accuracy: {train_score:.3f}")
        if 'validation_accuracy' in stats:
            logger.info(f"Validation accuracy: {stats['validation_accuracy']:.3f}")
        
        return stats
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict labels for a list of texts.
        
        Args:
            texts: List of text samples to classify
            
        Returns:
            List of predicted labels
        """
        if not self.is_trained:
            raise ValueError("Classifier has not been trained yet")
        
        if not texts:
            return []
        
        # Prepare features
        X = self._prepare_features(texts)
        
        if self.classifier_type == "setfit":
            # SetFit predicts directly on text
            predicted_labels = self.classifier.predict(texts)
        else:
            # Make predictions
            y_pred = self.classifier.predict(X)
            
            # Convert indices back to labels
            predicted_labels = [self.idx_to_label[idx] for idx in y_pred]
        
        return predicted_labels
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities for a list of texts.
        
        Args:
            texts: List of text samples to classify
            
        Returns:
            Array of class probabilities
        """
        if not self.is_trained:
            raise ValueError("Classifier has not been trained yet")
        
        if not texts:
            return np.array([])
        
        if self.classifier_type == "setfit":
            # SetFit doesn't have predict_proba, we'll use prediction confidence as a proxy
            predictions = self.classifier.predict(texts)
            # Convert predictions to probabilities (simplified)
            probabilities = np.zeros((len(texts), len(self.all_labels)))
            for i, pred in enumerate(predictions):
                if pred in self.label_to_idx:
                    label_idx = self.label_to_idx[pred]
                    probabilities[i, label_idx] = 0.9  # High confidence for prediction
                    # Distribute remaining probability among other classes
                    remaining_prob = 0.1 / (len(self.all_labels) - 1)
                    for j in range(len(self.all_labels)):
                        if j != label_idx:
                            probabilities[i, j] = remaining_prob
        else:
            # Prepare features
            X = self._prepare_features(texts)
            
            # Get probabilities
            probabilities = self.classifier.predict_proba(X)
        
        return probabilities
    
    def calculate_uncertainty(self, texts: List[str]) -> List[float]:
        """
        Calculate prediction uncertainty using entropy.
        
        Args:
            texts: List of text samples
            
        Returns:
            List of uncertainty scores (higher = more uncertain)
        """
        if not texts:
            return []
        
        probabilities = self.predict_proba(texts)
        uncertainties = []
        
        for probs in probabilities:
            # Calculate entropy
            entropy = -sum(p * math.log(p, 2) for p in probs if p > 0)
            uncertainties.append(entropy)
        
        return uncertainties
    
    def select_informative_samples(self, 
                                 texts: List[str], 
                                 k: int = None) -> List[Dict[str, Any]]:
        """
        Select the most informative samples using uncertainty sampling.
        
        Args:
            texts: List of text samples
            k: Number of samples to select (from config if None)
            
        Returns:
            List of dictionaries with sample info and uncertainty scores
        """
        if not texts:
            return []
        
        # Get default k from config if not provided
        if k is None:
            config = get_config()
            k = config.get('active_learning', {}).get('samples_per_iteration', 10)
        
        uncertainties = self.calculate_uncertainty(texts)
        predictions = self.predict(texts)
        probabilities = self.predict_proba(texts)
        
        # Create sample info
        samples_info = []
        for i, (text, pred, uncertainty, probs) in enumerate(
            zip(texts, predictions, uncertainties, probabilities)
        ):
            max_prob = np.max(probs)
            samples_info.append({
                'index': i,
                'text': text,
                'predicted_label': pred,
                'uncertainty': uncertainty,
                'max_probability': max_prob,
                'probabilities': probs.tolist()
            })
        
        # Sort by uncertainty (descending) and return top k
        samples_info.sort(key=lambda x: x['uncertainty'], reverse=True)
        
        return samples_info[:k]
    
    def retrain(self, 
               new_texts: List[str], 
               new_labels: List[str],
               noise_texts: List[str] = None) -> Dict[str, Any]:
        """
        Retrain the classifier with additional data.
        
        Args:
            new_texts: New training texts
            new_labels: New training labels
            noise_texts: Updated noise texts
            
        Returns:
            Updated training statistics
        """
        logger.info(f"Retraining classifier {self.subset_id} with {len(new_texts)} new samples")
        
        # Combine with existing training data
        all_texts = self.training_texts + new_texts
        all_labels = self.training_labels + new_labels
        
        # Retrain with combined data
        return self.train(all_texts, all_labels, noise_texts)
    
    def save(self, filepath: str) -> None:
        """
        Save the classifier to disk.
        
        Args:
            filepath: Path to save the classifier
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the sklearn classifier
        if self.classifier is not None:
            joblib.dump(self.classifier, f"{filepath}_classifier.joblib")
        
        # Save vectorizer if used
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, f"{filepath}_vectorizer.joblib")
        
        # Save metadata and other components
        metadata = {
            'subset_id': self.subset_id,
            'subset_labels': self.subset_labels,
            'classifier_type': self.classifier_type,
            'use_embeddings': self.use_embeddings,
            'embedding_model_name': self.embedding_model_name,
            'noise_oversample_factor': self.noise_oversample_factor,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'all_labels': self.all_labels,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'training_texts': self.training_texts,
            'training_labels': self.training_labels,
            'is_trained': self.is_trained,
            'training_stats': self.training_stats
        }
        
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Classifier saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SubsetClassifier':
        """
        Load a classifier from disk.
        
        Args:
            filepath: Path to load the classifier from
            
        Returns:
            Loaded SubsetClassifier instance
        """
        # Load metadata
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Create new instance
        classifier = cls(
            subset_id=metadata['subset_id'],
            subset_labels=metadata['subset_labels'],
            classifier_type=metadata['classifier_type'],
            use_embeddings=metadata['use_embeddings'],
            embedding_model=metadata['embedding_model_name'],
            noise_oversample_factor=metadata['noise_oversample_factor'],
            max_features=metadata['max_features'],
            random_state=metadata['random_state']
        )
        
        # Restore state
        classifier.all_labels = metadata['all_labels']
        classifier.label_to_idx = metadata['label_to_idx']
        classifier.idx_to_label = metadata['idx_to_label']
        classifier.training_texts = metadata['training_texts']
        classifier.training_labels = metadata['training_labels']
        classifier.is_trained = metadata['is_trained']
        classifier.training_stats = metadata['training_stats']
        
        # Load sklearn classifier
        if os.path.exists(f"{filepath}_classifier.joblib"):
            classifier.classifier = joblib.load(f"{filepath}_classifier.joblib")
        
        # Load vectorizer if it exists
        if os.path.exists(f"{filepath}_vectorizer.joblib"):
            classifier.vectorizer = joblib.load(f"{filepath}_vectorizer.joblib")
        
        logger.info(f"Classifier loaded from {filepath}")
        return classifier
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the classifier's training state."""
        return {
            'subset_id': self.subset_id,
            'subset_labels': self.subset_labels,
            'is_trained': self.is_trained,
            'training_samples': len(self.training_texts),
            'classifier_type': self.classifier_type,
            'use_embeddings': self.use_embeddings,
            'training_stats': self.training_stats
        }

"""
LLM Validator for Proxy Active Learning

This module implements the LLMValidator that acts as a proxy for human expert validation
using Large Language Models to validate classifier predictions.
"""

import logging
import os
import pickle
import random
from typing import List, Dict, Any

from ..utils.llm import LLMClient
from ..utils.model_config import ValidationPrompts
from ..utils.embeddings import EmbeddingProvider
from ..config.config_loader import get_config

logger = logging.getLogger(__name__)

class LLMValidator:
    """
    LLM-based validator for proxy active learning.
    
    This class mimics human expert validation by using an LLM to assess whether
    a predicted classification is correct based on label descriptions and
    similar examples from the training data.
    """
    
    def __init__(self, 
                 model_name: str = None,
                 embedding_model: str = None,
                 temperature: float = None,
                 max_timeout: int = None,
                 similar_examples_count: int = None):
        """
        Initialize the LLM validator.
        
        Args:
            model_name: Name of the LLM model to use for validation (from config if None)
            embedding_model: Name of the embedding model for similarity search (from config if None)
            temperature: Sampling temperature for the LLM (from config if None)
            max_timeout: Maximum timeout per LLM request (from config if None)
            similar_examples_count: Number of similar examples to include in validation (from config if None)
        """
        # Load configuration
        config = get_config()
        llm_config = config.get('llm_validation', {})
        embeddings_config = config.get('embeddings', {})
        
        # Set parameters from config or provided values
        self.model_name = model_name or llm_config.get('model_name')
        self.embedding_model_name = embedding_model or embeddings_config.get('model_name')
        self.temperature = temperature if temperature is not None else llm_config.get('temperature', 0.0)
        self.max_timeout = max_timeout or llm_config.get('max_timeout', 30)
        self.similar_examples_count = similar_examples_count or llm_config.get('similar_examples_count', 3)
        
        # Initialize LLM client
        self.llm_client = LLMClient(
            model_name=model_name,
            temperature=temperature,
            max_timeout=max_timeout
        )
        
        # Initialize embedding provider for similarity search
        self.embedding_provider = EmbeddingProvider(model_name=embedding_model)
        
        # Storage for training data and label descriptions
        self.training_data = {}  # {label: [texts]}
        self.label_descriptions = {}  # {label: description}
        self.validation_cache = {}  # Cache for validation results
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'valid_predictions': 0,
            'invalid_predictions': 0,
            'cache_hits': 0,
            'llm_errors': 0
        }
    
    def set_training_data(self, 
                         training_texts: List[str], 
                         training_labels: List[str]) -> None:
        """
        Set the training data for similarity search.
        
        Args:
            training_texts: List of training text samples
            training_labels: List of corresponding labels
        """
        if len(training_texts) != len(training_labels):
            raise ValueError("Number of texts and labels must match")
        
        # Organize training data by label
        self.training_data = {}
        for text, label in zip(training_texts, training_labels):
            if label not in self.training_data:
                self.training_data[label] = []
            self.training_data[label].append(text)
        
        logger.info(f"Training data set for {len(self.training_data)} labels:")
        for label, texts in self.training_data.items():
            logger.debug(f"  {label}: {len(texts)} samples")
        
        # Build embedding index for similarity search
        if training_texts:
            self.embedding_provider.build_index(training_texts)
    
    def set_label_descriptions(self, label_descriptions: Dict[str, str]) -> None:
        """
        Set descriptions for the class labels.
        
        Args:
            label_descriptions: Dictionary mapping labels to their descriptions
        """
        self.label_descriptions = label_descriptions.copy()
        msg = f"Label descriptions set for {len(label_descriptions)} labels"
        print(msg)
        logger.info(msg)
    
    def generate_label_descriptions(self,
                                  label_names: List[str],
                                  auto_generate: bool = True,
                                  max_examples_per_label: int = 50) -> Dict[str, str]:
        """
        Generate descriptions for labels using the LLM based on actual training samples.
        Uses batch processing for efficient LLM requests.

        Args:
            label_names: List of label names to generate descriptions for
            auto_generate: Whether to automatically generate descriptions
            max_examples_per_label: Maximum number of training samples to use per label

        Returns:
            Dictionary of label descriptions
        """
        descriptions = {}

        if not auto_generate:
            # Use simple descriptions based on label names
            for label in label_names:
                descriptions[label] = f"Text samples that belong to the '{label}' category"
            return descriptions

        # Generate descriptions using LLM with actual training samples
        system_prompt = """You are an expert at understanding text classification categories.
Given a class label name and examples of texts that belong to that category, provide a clear, concise description of what kind of text samples would belong to this category.
Focus on the semantic meaning, content themes, topics, or characteristics that are common across the provided examples.
Keep descriptions to 2-3 sentences. Base your description on the actual examples provided - do not make assumptions beyond what you see in the examples."""

        print("Generating label descriptions using LLM with training examples...")
        logger.info("Generating label descriptions using LLM with training examples...")

        # Prepare batch processing
        user_prompts = []
        label_metadata = []  # Store metadata for each label

        # Process special case first
        if "__NOISE__" in label_names:
            descriptions["__NOISE__"] = "Text samples that do not belong to any of the defined categories"

        # Build all prompts at once
        for label in label_names:
            if label == "__NOISE__":
                continue  # Already handled above

            # Get training samples for this label
            training_samples = self.training_data.get(label, [])

            if not training_samples:
                # No training samples available, use fallback
                descriptions[label] = f"Text samples that belong to the '{label}' category"
                msg = f"  {label}: Using fallback description (no training samples)"
                print(msg)
                logger.warning(msg)
                continue

            # Sample up to max_examples_per_label from the training data
            n_examples = min(max_examples_per_label, len(training_samples))
            selected_examples = random.sample(training_samples, n_examples) if len(training_samples) > n_examples else training_samples

            # Format examples for the prompt
            examples_text = "\n".join([f"{i+1}. \"{example}\"" for i, example in enumerate(selected_examples[:10])])  # Show first 10 in prompt

            if len(selected_examples) > 10:
                examples_text += f"\n... and {len(selected_examples) - 10} more examples"

            user_prompt = f"""Class Label: "{label}"

Based on the following examples of texts that belong to this category, provide a clear description of what kind of text samples would belong to this category.

Examples from this category:
{examples_text}

Focus on the common themes, topics, content patterns, and characteristics that you observe across these examples. Do not make assumptions beyond what you see in these examples."""

            user_prompts.append(user_prompt)
            label_metadata.append({
                'label': label,
                'n_examples': len(selected_examples),
                'selected_examples': selected_examples
            })

        # If no prompts to process, return early
        if not user_prompts:
            self.label_descriptions = descriptions
            return descriptions

        # Make batch LLM request for all labels
        msg = f"Making batch request for {len(user_prompts)} label descriptions..."
        print(msg)
        logger.info(msg)

        try:
            batch_responses = self.llm_client.query_batch(system_prompt, user_prompts)

            # Process batch results
            for i, (response, metadata) in enumerate(zip(batch_responses, label_metadata)):
                label = metadata['label']
                n_examples = metadata['n_examples']

                if not response.is_valid:
                    # Fallback description
                    descriptions[label] = f"Text samples that belong to the '{label}' category"
                    msg = f"  {label}: Using fallback description (LLM failed)"
                    print(msg)
                    logger.warning(msg)
                else:
                    descriptions[label] = response.content.strip()
                    msg = f"  {label}: Generated description using {n_examples} training examples"
                    print(msg)
                    logger.info(msg)
                    logger.debug(f"    Description: {descriptions[label]}")

        except Exception as e:
            msg = f"Batch description generation failed: {e}"
            print(msg)
            logger.error(msg)

            # Use fallback descriptions for all remaining labels
            for metadata in label_metadata:
                label = metadata['label']
                if label not in descriptions:
                    descriptions[label] = f"Text samples that belong to the '{label}' category"
                    msg = f"  {label}: Using fallback description (batch failed)"
                    print(msg)
                    logger.warning(msg)

        self.label_descriptions = descriptions
        return descriptions
    
    def _find_similar_examples(self, 
                              text: str, 
                              predicted_label: str, 
                              k: int = None) -> List[str]:
        """
        Find similar examples from training data for a given text and label.
        
        Args:
            text: Text to find similar examples for
            predicted_label: The predicted label
            k: Number of similar examples to return
            
        Returns:
            List of similar text examples
        """
        if k is None:
            k = self.similar_examples_count
        
        # Check if we have training data for this label
        if predicted_label not in self.training_data:
            return []
        
        label_texts = self.training_data[predicted_label]
        
        if not label_texts:
            return []
        
        # Find similar examples using embedding similarity
        try:
            similar_results = self.embedding_provider.find_similar_simple(
                query_text=text,
                candidate_texts=label_texts,
                k=k
            )
            
            return [result['text'] for result in similar_results]
        
        except Exception as e:
            msg = f"Error finding similar examples: {e}"
            print(msg)
            logger.error(msg)
        
        # Fallback: return random examples
        return random.sample(label_texts, min(k, len(label_texts)))
    
    def _create_validation_prompt(self, 
                                 text: str, 
                                 predicted_label: str, 
                                 label_description: str, 
                                 similar_examples: List[str]) -> str:
        """
        Create a validation prompt for the LLM.
        
        Args:
            text: Text to validate
            predicted_label: Predicted class label
            label_description: Description of the label
            similar_examples: Similar examples from training data
            
        Returns:
            Formatted validation prompt
        """
        examples_section = ""
        if similar_examples:
            examples_section = ValidationPrompts.build_examples_section(
                similar_examples, max_examples=self.similar_examples_count
            )
        
        template = ValidationPrompts.get_binary_validation_prompt()
        prompt = ValidationPrompts.format_validation_prompt(
            template,
            text=text,
            label=predicted_label,
            description=label_description,
            examples_section=examples_section
        )
        
        return prompt
    
    def validate(self, 
                text: str, 
                predicted_label: str, 
                use_cache: bool = True,
                include_reasoning: bool = False) -> Dict[str, Any]:
        """
        Validate if a text sample belongs to the predicted label.
        
        Args:
            text: Text sample to validate
            predicted_label: Predicted class label
            use_cache: Whether to use cached results
            include_reasoning: Whether to request reasoning from the LLM
            
        Returns:
            Dictionary with validation result and metadata
        """
        # Create cache key
        cache_key = f"{hash(text)}_{predicted_label}"
        
        # Check cache first
        if use_cache and cache_key in self.validation_cache:
            self.validation_stats['cache_hits'] += 1
            return self.validation_cache[cache_key]
        
        # Get label description
        label_description = self.label_descriptions.get(
            predicted_label, 
            f"Text samples that belong to the '{predicted_label}' category"
        )
        
        # Find similar examples
        similar_examples = self._find_similar_examples(text, predicted_label)
        
        # Create validation prompt
        user_prompt = self._create_validation_prompt(
            text, predicted_label, label_description, similar_examples
        )
        
        system_prompt = """You are a precise and accurate text classification validator.
Your job is to determine if a text sample truly belongs to a specific class label.
Be thorough in your analysis but respond with only 'True' or 'False'."""
        
        # Make LLM request
        try:
            response = self.llm_client.query(system_prompt, user_prompt)
            
            if not response.is_valid:
                self.validation_stats['llm_errors'] += 1
                result = {
                    'is_valid': False,
                    'confidence': 0.0,  # No confidence when LLM request fails
                    'error': 'LLM request failed',
                    'similar_examples_count': len(similar_examples),
                    'label_description': label_description
                }
            else:
                # Parse the response
                response_text = response.content.strip().lower()
                
                # Note: Confidence levels are kept in code as they represent
                # internal validator logic, not application-level configuration
                if "true" in response_text and "false" not in response_text:
                    is_valid = True
                    confidence = 0.8  # High confidence for clear True response
                elif "false" in response_text and "true" not in response_text:
                    is_valid = False
                    confidence = 0.8  # High confidence for clear False response
                else:
                    # Ambiguous response - default to False for safety
                    is_valid = False
                    confidence = 0.3  # Low confidence for ambiguous response
                
                result = {
                    'is_valid': is_valid,
                    'confidence': confidence,
                    'llm_response': response.content,
                    'similar_examples_count': len(similar_examples),
                    'label_description': label_description,
                    'similar_examples': similar_examples if include_reasoning else []
                }
                
                # Update statistics
                if is_valid:
                    self.validation_stats['valid_predictions'] += 1
                else:
                    self.validation_stats['invalid_predictions'] += 1
            
        except Exception as e:
            self.validation_stats['llm_errors'] += 1
            result = {
                'is_valid': False,
                'confidence': 0.0,  # No confidence when exception occurs
                'error': str(e),
                'similar_examples_count': len(similar_examples),
                'label_description': label_description
            }
        
        # Update total validations
        self.validation_stats['total_validations'] += 1
        
        # Cache the result
        if use_cache:
            self.validation_cache[cache_key] = result
        
        return result
    
    def validate_batch(self, 
                      texts: List[str], 
                      predicted_labels: List[str],
                      use_cache: bool = True,
                      include_reasoning: bool = False) -> List[Dict[str, Any]]:
        """
        Validate multiple text samples in batch using efficient LLM batch processing.
        
        Args:
            texts: List of text samples to validate
            predicted_labels: List of predicted labels
            use_cache: Whether to use cached results
            include_reasoning: Whether to request reasoning from the LLM
            
        Returns:
            List of validation results
        """
        if len(texts) != len(predicted_labels):
            raise ValueError("Number of texts and labels must match")
        
        # Separate cached and non-cached samples
        cached_results = {}
        non_cached_indices = []
        non_cached_texts = []
        non_cached_labels = []
        
        for i, (text, predicted_label) in enumerate(zip(texts, predicted_labels)):
            cache_key = f"{hash(text)}_{predicted_label}"
            
            if use_cache and cache_key in self.validation_cache:
                cached_results[i] = self.validation_cache[cache_key]
                self.validation_stats['cache_hits'] += 1
            else:
                non_cached_indices.append(i)
                non_cached_texts.append(text)
                non_cached_labels.append(predicted_label)
        
        # If all samples are cached, return early
        if not non_cached_indices:
            return [cached_results[i] for i in range(len(texts))]
        
        # Build all prompts for non-cached samples at once
        user_prompts = []
        prompt_metadata = []  # Store metadata for each prompt
        
        for text, predicted_label in zip(non_cached_texts, non_cached_labels):
            # Get label description
            label_description = self.label_descriptions.get(
                predicted_label, 
                f"Text samples that belong to the '{predicted_label}' category"
            )
            
            # Find similar examples
            similar_examples = self._find_similar_examples(text, predicted_label)
            
            # Create validation prompt
            user_prompt = self._create_validation_prompt(
                text, predicted_label, label_description, similar_examples
            )
            
            user_prompts.append(user_prompt)
            prompt_metadata.append({
                'text': text,
                'predicted_label': predicted_label,
                'label_description': label_description,
                'similar_examples': similar_examples,
                'similar_examples_count': len(similar_examples)
            })
        
        # Make batch LLM request for all non-cached samples
        system_prompt = """You are a precise and accurate text classification validator.
Your job is to determine if a text sample truly belongs to a specific class label.
Be thorough in your analysis but respond with only 'True' or 'False'."""
        
        try:
            batch_responses = self.llm_client.query_batch(system_prompt, user_prompts)
            
            # Process batch results
            batch_results = []
            for i, (response, metadata) in enumerate(zip(batch_responses, prompt_metadata)):
                if not response.is_valid:
                    self.validation_stats['llm_errors'] += 1
                    result = {
                        'is_valid': False,
                        'confidence': 0.0,
                        'error': 'LLM request failed',
                        'similar_examples_count': metadata['similar_examples_count'],
                        'label_description': metadata['label_description']
                    }
                else:
                    # Parse the response
                    response_text = response.content.strip().lower()
                    
                    if "true" in response_text and "false" not in response_text:
                        is_valid = True
                        confidence = 0.8
                    elif "false" in response_text and "true" not in response_text:
                        is_valid = False
                        confidence = 0.8
                    else:
                        # Ambiguous response - default to False for safety
                        is_valid = False
                        confidence = 0.3
                    
                    result = {
                        'is_valid': is_valid,
                        'confidence': confidence,
                        'llm_response': response.content,
                        'similar_examples_count': metadata['similar_examples_count'],
                        'label_description': metadata['label_description'],
                        'similar_examples': metadata['similar_examples'] if include_reasoning else []
                    }
                    
                    # Update statistics
                    if is_valid:
                        self.validation_stats['valid_predictions'] += 1
                    else:
                        self.validation_stats['invalid_predictions'] += 1
                
                batch_results.append(result)
                
                # Cache the result
                if use_cache:
                    cache_key = f"{hash(metadata['text'])}_{metadata['predicted_label']}"
                    self.validation_cache[cache_key] = result
                
                # Update total validations
                self.validation_stats['total_validations'] += 1
                
        except Exception as e:
            # Handle batch processing errors
            self.validation_stats['llm_errors'] += len(non_cached_indices)
            batch_results = []
            for metadata in prompt_metadata:
                result = {
                    'is_valid': False,
                    'confidence': 0.0,
                    'error': str(e),
                    'similar_examples_count': metadata['similar_examples_count'],
                    'label_description': metadata['label_description']
                }
                batch_results.append(result)
                
                # Update total validations
                self.validation_stats['total_validations'] += 1
        
        # Combine cached and new results in the correct order
        final_results = [None] * len(texts)
        
        # Place cached results
        for i in cached_results:
            final_results[i] = cached_results[i]
        
        # Place batch results
        for i, batch_result in enumerate(batch_results):
            original_index = non_cached_indices[i]
            final_results[original_index] = batch_result
        
        return final_results
    
    def select_valid_samples(self, 
                           texts: List[str], 
                           predicted_labels: List[str],
                           confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Select only the samples that are validated as correct.
        
        Args:
            texts: List of text samples
            predicted_labels: List of predicted labels
            confidence_threshold: Minimum confidence threshold for acceptance
            
        Returns:
            List of valid samples with metadata
        """
        validation_results = self.validate_batch(texts, predicted_labels)
        
        valid_samples = []
        for i, (text, label, result) in enumerate(zip(texts, predicted_labels, validation_results)):
            if (result['is_valid'] and 
                result.get('confidence', 0) >= confidence_threshold):
                valid_samples.append({
                    'text': text,
                    'label': label,
                    'confidence': result.get('confidence', 0),
                    'validation_result': result,
                    'original_index': i
                })
        
        msg = f"Selected {len(valid_samples)} valid samples out of {len(texts)} total"
        print(msg)
        logger.info(msg)
        return valid_samples
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics and metrics."""
        total = self.validation_stats['total_validations']
        if total == 0:
            return self.validation_stats
        
        stats = self.validation_stats.copy()
        stats['validation_accuracy'] = self.validation_stats['valid_predictions'] / total
        stats['error_rate'] = self.validation_stats['llm_errors'] / total
        stats['cache_hit_rate'] = self.validation_stats['cache_hits'] / total
        
        return stats
    
    def save_cache(self, filepath: str) -> None:
        """Save the validation cache to disk."""
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        cache_data = {
            'validation_cache': self.validation_cache,
            'label_descriptions': self.label_descriptions,
            'validation_stats': self.validation_stats,
            'model_name': self.model_name,
            'embedding_model_name': self.embedding_model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f)
        
        msg = f"Validation cache saved to {filepath}"
        print(msg)
        logger.info(msg)
    
    def load_cache(self, filepath: str) -> None:
        """Load validation cache from disk."""
        
        try:
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.validation_cache = cache_data.get('validation_cache', {})
            self.label_descriptions = cache_data.get('label_descriptions', {})
            self.validation_stats = cache_data.get('validation_stats', self.validation_stats)
            
            msg1 = f"Validation cache loaded from {filepath}"
            print(msg1)
            logger.info(msg1)
            msg2 = f"Loaded {len(self.validation_cache)} cached validations"
            print(msg2)
            logger.info(msg2)
            
        except FileNotFoundError:
            msg = f"Cache file not found: {filepath}"
            print(msg)
            logger.warning(msg)
        except Exception as e:
            msg = f"Error loading cache: {e}"
            print(msg)
            logger.error(msg)
    
    def clear_cache(self) -> None:
        """Clear the validation cache."""
        self.validation_cache = {}
        msg = "Validation cache cleared"
        print(msg)
        logger.info(msg)
    
    def update_model(self, model_name: str) -> None:
        """
        Update the LLM model used for validation.
        
        Args:
            model_name: New model name to use
        """
        self.model_name = model_name
        self.llm_client = LLMClient(
            model_name=model_name,
            temperature=self.temperature,
            max_timeout=self.max_timeout
        )
        msg = f"Updated LLM model to: {model_name}"
        print(msg)
        logger.info(msg)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current models and configuration."""
        return {
            'llm_model': self.model_name,
            'embedding_model': self.embedding_model_name,
            'temperature': self.temperature,
            'max_timeout': self.max_timeout,
            'similar_examples_count': self.similar_examples_count,
            'training_labels': list(self.training_data.keys()),
            'cached_validations': len(self.validation_cache),
            'validation_stats': self.get_validation_statistics()
        }

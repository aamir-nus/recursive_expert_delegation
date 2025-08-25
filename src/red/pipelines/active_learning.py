"""
Active Learning Loop for R.E.D. Framework

This module implements the main recursive active learning loop that classifies
unlabeled data, validates predictions with LLM, and retrains classifiers.
"""
import time
import json
from typing import Dict, List, Any
from pathlib import Path
import numpy as np

from ..core.subsetter import LabelSubsetter
from ..core.classifier import SubsetClassifier
from ..core.validator import LLMValidator
from ..data.data_manager import DataManager
from ..config.config_loader import get_config

class ActiveLearningLoop:
    """
    Main recursive engine of the R.E.D. framework.
    
    This class orchestrates the active learning process:
    1. Loads unlabeled data in batches
    2. Makes preliminary predictions using subset classifiers
    3. Validates predictions using LLM
    4. Adds validated samples to training data
    5. Retrains classifiers when thresholds are met
    6. Continues until convergence or maximum iterations
    """
    
    def __init__(self, 
                 components_dir: str,
                 output_dir: str = None,
                 config_dir: str = None):
        """
        Initialize the active learning loop.
        
        Args:
            components_dir: Directory containing trained components
            output_dir: Directory to save outputs
            config_dir: Directory containing configuration files
        """
        # Load configuration
        self.config = get_config()
        
        # Set directories
        self.components_dir = Path(components_dir)
        self.output_dir = Path(output_dir or self.config.get('data', {}).get('output_dir', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration shortcuts
        self.active_learning_config = self.config.get('active_learning', {})
        self.llm_config = self.config.get('llm_validation', {})
        
        # Initialize components (will be loaded)
        self.data_manager = None
        self.label_subsetter = None
        self.subset_classifiers = {}
        self.llm_validator = None
        self.subset_mapping = {}
        
        # Active learning state
        self.current_iteration = 0
        self.total_validated_samples = 0
        self.validation_history = []
        self.retrain_history = []
        self.convergence_achieved = False
        
        # Performance tracking
        self.iteration_stats = []
        self.processing_times = []
        
    def run(self, 
            unlabeled_data_path: str,
            max_iterations: int = None,
            batch_size: int = None,
            samples_per_iteration: int = None) -> Dict[str, Any]:
        """
        Run the complete active learning loop.
        
        Args:
            unlabeled_data_path: Path to unlabeled data file
            max_iterations: Maximum number of iterations
            batch_size: Size of data batches to process
            samples_per_iteration: Number of samples to validate per iteration
            
        Returns:
            Dictionary with results and statistics
        """
        print("=" * 60)
        print("R.E.D. ACTIVE LEARNING LOOP")
        print("=" * 60)
        
        start_time = time.time()
        
        # Set parameters
        max_iterations = max_iterations or self.active_learning_config.get('max_iterations', 10)
        batch_size = batch_size or self.active_learning_config.get('batch_size', 100)
        samples_per_iteration = samples_per_iteration or self.active_learning_config.get('samples_per_iteration', 50)
        
        try:
            # Step 1: Load trained components
            print("\n" + "="*50)
            print("STEP 1: LOADING TRAINED COMPONENTS")
            print("="*50)
            self._load_components()
            
            # Step 2: Load unlabeled data
            print("\n" + "="*50)
            print("STEP 2: LOADING UNLABELED DATA")
            print("="*50)
            self._load_unlabeled_data(unlabeled_data_path)
            
            # Step 3: Run active learning iterations
            print("\n" + "="*50)
            print("STEP 3: RUNNING ACTIVE LEARNING ITERATIONS")
            print("="*50)
            
            iteration_results = []
            
            for iteration in range(max_iterations):
                self.current_iteration = iteration + 1
                
                print(f"\n{'='*20} ITERATION {self.current_iteration} {'='*20}")
                
                # Run single iteration
                iter_result = self._run_iteration(
                    batch_size=batch_size,
                    samples_per_iteration=samples_per_iteration
                )
                
                iteration_results.append(iter_result)
                
                # Check convergence
                if self._check_convergence(iter_result):
                    print(f"\n✓ Convergence achieved after {self.current_iteration} iterations")
                    self.convergence_achieved = True
                    break
                
                print(f"Iteration {self.current_iteration} completed. "
                      f"Validated: {iter_result['validated_samples']}, "
                      f"Time: {iter_result['iteration_time']:.2f}s")
            
            # Step 4: Final retraining
            print("\n" + "="*50)
            print("STEP 4: FINAL RETRAINING")
            print("="*50)
            final_retrain_stats = self._final_retrain()
            
            # Step 5: Save final state
            print("\n" + "="*50)
            print("STEP 5: SAVING FINAL STATE")
            print("="*50)
            save_stats = self._save_final_state()
            
            # Compile results
            total_time = time.time() - start_time
            
            results = {
                'status': 'success',
                'total_time': total_time,
                'total_iterations': self.current_iteration,
                'convergence_achieved': self.convergence_achieved,
                'total_validated_samples': self.total_validated_samples,
                'iteration_results': iteration_results,
                'final_retrain_stats': final_retrain_stats,
                'save_stats': save_stats,
                'validation_history': self.validation_history,
                'retrain_history': self.retrain_history,
                'performance_stats': self._get_performance_stats(),
                'output_dir': str(self.output_dir)
            }
            
            # Save results summary
            self._save_results_summary(results)
            
            print("\n" + "="*60)
            print("ACTIVE LEARNING COMPLETED SUCCESSFULLY")
            print(f"Total iterations: {self.current_iteration}")
            print(f"Total validated samples: {self.total_validated_samples}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Output directory: {self.output_dir}")
            print("="*60)
            
            return results
            
        except Exception as e:
            print(f"\nERROR during active learning: {e}")
            error_results = {
                'status': 'error',
                'error': str(e),
                'current_iteration': self.current_iteration,
                'total_time': time.time() - start_time
            }
            self._save_results_summary(error_results)
            raise
    
    def _load_components(self) -> None:
        """Load all trained components from disk."""
        print(f"Loading components from: {self.components_dir}")
        
        # Initialize data manager (will use config internally)
        self.data_manager = DataManager(
            data_dir=str(self.components_dir.parent / "data")
        )
        
        # Load data manager state
        data_manager_path = self.components_dir / "data_manager.pkl"
        if data_manager_path.exists():
            self.data_manager.load_state(str(data_manager_path))
            print("✓ Loaded data manager state")
        
        # Load subset mapping
        mapping_path = self.components_dir / "subset_mapping.json"
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                self.subset_mapping = json.load(f)
            self.data_manager.set_subset_mapping(self.subset_mapping)
            print(f"✓ Loaded subset mapping ({len(self.subset_mapping)} subsets)")
        else:
            raise FileNotFoundError(f"Subset mapping not found: {mapping_path}")
        
        # Load label subsetter
        subsetter_path = self.components_dir / "label_subsetter.pkl"
        if subsetter_path.exists():
            self.label_subsetter = LabelSubsetter.load(str(subsetter_path))
            print("✓ Loaded label subsetter")
        
        # Load subset classifiers
        classifiers_dir = self.components_dir / "classifiers"
        if classifiers_dir.exists():
            for subset_id in self.subset_mapping.keys():
                classifier_path = classifiers_dir / f"{subset_id}"
                try:
                    self.subset_classifiers[subset_id] = SubsetClassifier.load(str(classifier_path))
                    print(f"✓ Loaded classifier for {subset_id}")
                except Exception as e:
                    print(f"⚠ Failed to load classifier for {subset_id}: {e}")
        
        print(f"✓ Loaded {len(self.subset_classifiers)} subset classifiers")
        
        # Initialize LLM validator (will use config internally)
        self.llm_validator = LLMValidator()
        
        # Set training data for LLM validator
        self.llm_validator.set_training_data(
            training_texts=self.data_manager.seed_data['texts'],
            training_labels=self.data_manager.seed_data['labels']
        )
        
        # Load validator cache
        validator_cache_path = self.components_dir / "validator_cache.pkl"
        if validator_cache_path.exists():
            self.llm_validator.load_cache(str(validator_cache_path))
            print("✓ Loaded validator cache")
        
        # Generate label descriptions if needed
        unique_labels = list(set(self.data_manager.seed_data['labels']))
        self.llm_validator.generate_label_descriptions(
            label_names=unique_labels,
            auto_generate=self.llm_config.get('auto_generate_descriptions', True)
        )
        
        print("✓ LLM validator initialized")
    
    def _load_unlabeled_data(self, unlabeled_data_path: str) -> None:
        """Load unlabeled data for processing."""
        print(f"Loading unlabeled data from: {unlabeled_data_path}")
        
        num_samples = self.data_manager.load_unlabeled_data(
            filepath=unlabeled_data_path,
            text_column=self.config.get('data', {}).get('text_column', 'text')
        )
        
        print(f"✓ Loaded {num_samples} unlabeled samples")
    
    def _run_iteration(self, batch_size: int, samples_per_iteration: int) -> Dict[str, Any]:
        """Run a single active learning iteration."""
        iteration_start = time.time()
        
        # Get batch of unlabeled data
        start_idx = (self.current_iteration - 1) * batch_size
        batch_texts = self.data_manager.get_unlabeled_batch(
            batch_size=batch_size,
            start_index=start_idx
        )
        
        if not batch_texts:
            print("No more unlabeled data to process")
            return {
                'iteration': self.current_iteration,
                'batch_size': 0,
                'predictions_made': 0,
                'validated_samples': 0,
                'retrained_classifiers': 0,
                'iteration_time': time.time() - iteration_start
            }
        
        print(f"Processing batch of {len(batch_texts)} samples...")
        
        # Make predictions for each subset
        all_predictions = []
        prediction_metadata = []
        
        for subset_id, classifier in self.subset_classifiers.items():
            # Get predictions and uncertainties
            predictions = classifier.predict(batch_texts)
            uncertainties = classifier.calculate_uncertainty(batch_texts)
            probabilities = classifier.predict_proba(batch_texts)
            
            for i, (text, pred, uncertainty, probs) in enumerate(
                zip(batch_texts, predictions, uncertainties, probabilities)
            ):
                if pred != "__NOISE__":  # Only consider non-noise predictions
                    all_predictions.append({
                        'text': text,
                        'predicted_label': pred,
                        'subset_id': subset_id,
                        'uncertainty': uncertainty,
                        'max_probability': np.max(probs),
                        'classifier': classifier,
                        'batch_index': start_idx + i
                    })
        
        print(f"Made {len(all_predictions)} non-noise predictions")
        
        # Select most informative samples
        informative_samples = self._select_informative_samples(
            all_predictions, samples_per_iteration
        )
        
        print(f"Selected {len(informative_samples)} informative samples for validation")
        
        # Validate selected samples
        validated_samples = self._validate_samples(informative_samples)
        
        # Add validated samples to training data
        new_training_samples = 0
        for sample in validated_samples:
            if sample['validation_result']['is_valid']:
                self.data_manager.add_validated_sample(
                    text=sample['text'],
                    label=sample['predicted_label'],
                    confidence=sample['validation_result'].get('confidence', 1.0),
                    validation_metadata=sample['validation_result']
                )
                new_training_samples += 1
        
        self.total_validated_samples += new_training_samples
        
        # Check if retraining is needed
        retrained_classifiers = 0
        retrain_threshold = self.active_learning_config.get('retrain_threshold', 100)
        
        if new_training_samples > 0 and self.total_validated_samples % retrain_threshold == 0:
            retrained_classifiers = self._retrain_classifiers()
        
        iteration_time = time.time() - iteration_start
        self.processing_times.append(iteration_time)
        
        # Store iteration statistics
        iter_stats = {
            'iteration': self.current_iteration,
            'batch_size': len(batch_texts),
            'predictions_made': len(all_predictions),
            'informative_samples': len(informative_samples),
            'validated_samples': new_training_samples,
            'retrained_classifiers': retrained_classifiers,
            'iteration_time': iteration_time,
            'cumulative_validated': self.total_validated_samples
        }
        
        self.iteration_stats.append(iter_stats)
        
        return iter_stats
    
    def _select_informative_samples(self, 
                                  predictions: List[Dict], 
                                  k: int) -> List[Dict]:
        """Select the most informative samples for validation."""
        if len(predictions) <= k:
            return predictions
        
        # Sort by uncertainty (higher uncertainty = more informative)
        predictions_sorted = sorted(predictions, key=lambda x: x['uncertainty'], reverse=True)
        
        # Take top k most uncertain predictions
        return predictions_sorted[:k]
    
    def _validate_samples(self, samples: List[Dict]) -> List[Dict]:
        """Validate samples using LLM validator with efficient batch processing."""
        print(f"Validating {len(samples)} samples using LLM batch processing...")
        
        # Extract texts and predicted labels for batch processing
        texts = [sample['text'] for sample in samples]
        predicted_labels = [sample['predicted_label'] for sample in samples]
        
        try:
            # Use batch validation for efficiency
            validation_results = self.llm_validator.validate_batch(
                texts=texts,
                predicted_labels=predicted_labels,
                use_cache=self.llm_config.get('use_cache', True)
            )
            
            # Attach validation results to samples
            validated_samples = []
            for sample, validation_result in zip(samples, validation_results):
                sample['validation_result'] = validation_result
                validated_samples.append(sample)
                
        except Exception as e:
            print(f"⚠ Batch validation failed: {e}")
            # Fallback to individual validation with error results
            validated_samples = []
            for sample in samples:
                sample['validation_result'] = {
                    'is_valid': False,
                    'confidence': 0.0,
                    'error': str(e)
                }
                validated_samples.append(sample)
        
        # Count valid predictions
        valid_count = sum(1 for s in validated_samples if s['validation_result']['is_valid'])
        print(f"✓ {valid_count}/{len(validated_samples)} samples validated as correct")
        
        # Store validation history
        self.validation_history.append({
            'iteration': self.current_iteration,
            'total_validated': len(validated_samples),
            'valid_samples': valid_count,
            'validation_rate': valid_count / len(validated_samples) if validated_samples else 0
        })
        
        return validated_samples
    
    def _retrain_classifiers(self) -> int:
        """Retrain classifiers with new validated data."""
        print("Retraining classifiers with new validated data...")
        
        retrain_all = self.active_learning_config.get('retrain_all', False)
        retrained_count = 0
        
        # Determine which classifiers to retrain
        if retrain_all:
            classifiers_to_retrain = list(self.subset_classifiers.keys())
        else:
            # Only retrain classifiers that have new data
            classifiers_to_retrain = []
            for subset_id in self.subset_classifiers.keys():
                subset_data = self.data_manager.get_subset_data(subset_id)
                if len(subset_data['texts']) > self.subset_classifiers[subset_id].training_stats.get('train_samples', 0):
                    classifiers_to_retrain.append(subset_id)
        
        for subset_id in classifiers_to_retrain:
            try:
                print(f"  Retraining {subset_id}...")
                
                # Get updated training data
                subset_data = self.data_manager.get_subset_data(subset_id)
                noise_data = self.data_manager.get_noise_data(subset_id, max_samples=1000)
                
                # Retrain classifier
                self.subset_classifiers[subset_id].retrain(
                    new_texts=subset_data['texts'],
                    new_labels=subset_data['labels'],
                    noise_texts=noise_data
                )
                
                retrained_count += 1
                print(f"  ✓ {subset_id} retrained successfully")
                
            except Exception as e:
                print(f"  ✗ Failed to retrain {subset_id}: {e}")
        
        if retrained_count > 0:
            self.retrain_history.append({
                'iteration': self.current_iteration,
                'retrained_classifiers': retrained_count,
                'total_classifiers': len(self.subset_classifiers)
            })
            
            print(f"✓ Retrained {retrained_count} classifiers")
        
        return retrained_count
    
    def _check_convergence(self, iter_result: Dict) -> bool:
        """Check if the active learning process has converged."""
        min_gain = self.active_learning_config.get('min_information_gain', 0.01)
        
        # Check if we're getting diminishing returns
        if len(self.validation_history) >= 3:
            recent_rates = [h['validation_rate'] for h in self.validation_history[-3:]]
            if all(rate < min_gain for rate in recent_rates):
                return True
        
        # Check if no new validated samples
        if iter_result['validated_samples'] == 0:
            return True
        
        return False
    
    def _final_retrain(self) -> Dict[str, Any]:
        """Perform final retraining of all classifiers."""
        print("Performing final retraining of all classifiers...")
        
        retrain_stats = {
            'total_classifiers': len(self.subset_classifiers),
            'successfully_retrained': 0,
            'failed_retrains': 0,
            'individual_stats': {}
        }
        
        for subset_id, classifier in self.subset_classifiers.items():
            try:
                print(f"  Final retrain for {subset_id}...")
                
                # Get all training data
                subset_data = self.data_manager.get_subset_data(subset_id)
                noise_data = self.data_manager.get_noise_data(subset_id, max_samples=1000)
                
                # Final retrain
                training_stats = classifier.retrain(
                    new_texts=subset_data['texts'],
                    new_labels=subset_data['labels'],
                    noise_texts=noise_data
                )
                
                retrain_stats['individual_stats'][subset_id] = training_stats
                retrain_stats['successfully_retrained'] += 1
                
                print(f"  ✓ {subset_id}: Final accuracy = {training_stats.get('train_accuracy', 'N/A')}")
                
            except Exception as e:
                print(f"  ✗ Failed final retrain for {subset_id}: {e}")
                retrain_stats['failed_retrains'] += 1
        
        print(f"✓ Final retraining completed: {retrain_stats['successfully_retrained']}/{retrain_stats['total_classifiers']} successful")
        
        return retrain_stats
    
    def _save_final_state(self) -> Dict[str, Any]:
        """Save the final state of all components."""
        print("Saving final state...")
        
        # Create final state directory
        final_state_dir = self.output_dir / "final_state"
        final_state_dir.mkdir(exist_ok=True)
        
        classifiers_dir = final_state_dir / "classifiers"
        classifiers_dir.mkdir(exist_ok=True)
        
        save_stats = {
            'saved_components': [],
            'failed_saves': [],
            'output_paths': {}
        }
        
        try:
            # Save updated classifiers
            for subset_id, classifier in self.subset_classifiers.items():
                classifier_path = classifiers_dir / f"{subset_id}"
                classifier.save(str(classifier_path))
                save_stats['saved_components'].append(f'classifier_{subset_id}')
                save_stats['output_paths'][f'classifier_{subset_id}'] = str(classifier_path)
            
            # Save updated data manager state
            data_manager_path = final_state_dir / "data_manager.pkl"
            self.data_manager.save_state(str(data_manager_path))
            save_stats['saved_components'].append('data_manager')
            save_stats['output_paths']['data_manager'] = str(data_manager_path)
            
            # Save LLM validator cache
            validator_cache_path = final_state_dir / "validator_cache.pkl"
            self.llm_validator.save_cache(str(validator_cache_path))
            save_stats['saved_components'].append('validator_cache')
            save_stats['output_paths']['validator_cache'] = str(validator_cache_path)
            
            # Export final training data
            training_data_path = final_state_dir / "final_training_data.csv"
            self.data_manager.export_training_data(
                filepath=str(training_data_path),
                include_validated=True,
                format='csv'
            )
            save_stats['saved_components'].append('training_data')
            save_stats['output_paths']['training_data'] = str(training_data_path)
            
            print(f"✓ Saved {len(save_stats['saved_components'])} components to final state")
            
        except Exception as e:
            print(f"✗ Error saving final state: {e}")
            save_stats['failed_saves'].append(str(e))
        
        return save_stats
    
    def _get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the active learning process."""
        if not self.processing_times:
            return {}
        
        return {
            'total_processing_time': sum(self.processing_times),
            'average_iteration_time': np.mean(self.processing_times),
            'min_iteration_time': min(self.processing_times),
            'max_iteration_time': max(self.processing_times),
            'total_iterations': len(self.processing_times),
            'validation_efficiency': self.total_validated_samples / sum(self.processing_times) if self.processing_times else 0
        }
    
    def _save_results_summary(self, results: Dict[str, Any]) -> None:
        """Save active learning results summary."""
        summary_path = self.output_dir / "active_learning_summary.json"
        
        try:
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"✓ Results summary saved to: {summary_path}")
        except Exception as e:
            print(f"⚠ Failed to save results summary: {e}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the active learning process."""
        return {
            'current_iteration': self.current_iteration,
            'total_validated_samples': self.total_validated_samples,
            'convergence_achieved': self.convergence_achieved,
            'validation_history': self.validation_history,
            'retrain_history': self.retrain_history,
            'iteration_stats': self.iteration_stats,
            'performance_stats': self._get_performance_stats()
        }

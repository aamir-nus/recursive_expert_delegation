"""
Initial Training Pipeline for R.E.D. Framework

This module orchestrates the one-time setup and initial training of the R.E.D. system,
including subset creation and initial classifier training.
"""
import time
from typing import Dict, Any
from pathlib import Path
import json

from ..core.subsetter import LabelSubsetter
from ..core.classifier import SubsetClassifier
from ..core.validator import LLMValidator
from ..data.data_manager import DataManager
from ..config.config_loader import get_config

class InitialTrainingPipeline:
    """
    Orchestrates the initial setup and training of the R.E.D. system.
    
    This pipeline:
    1. Loads seed training data
    2. Creates label subsets using greedy selection
    3. Trains subset classifiers with noise oversampling
    4. Sets up LLM validator with label descriptions
    5. Saves all trained components
    """
    
    def __init__(self, 
                 config_dir: str = None,
                 output_dir: str = None,
                 seed_data_path: str = None):
        """
        Initialize the initial training pipeline.
        
        Args:
            config_dir: Directory containing configuration files
            output_dir: Directory to save outputs
            seed_data_path: Path to seed training data
        """
        # Load configuration
        self.config = get_config()
        
        # Set directories
        self.output_dir = Path(output_dir or self.config.get('data', {}).get('output_dir', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = Path(self.config.get('data', {}).get('data_dir', './data'))
        self.log_dir = Path(self.config.get('data', {}).get('log_dir', './logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components (will use config internally)
        self.data_manager = DataManager(
            data_dir=str(self.data_dir)
        )
        
        self.label_subsetter = None
        self.subset_classifiers = {}
        self.llm_validator = None
        
        # Training state
        self.subset_mapping = {}
        self.training_stats = {}
        self.seed_data_path = seed_data_path
        
        # Configuration shortcuts
        self.subsetting_config = self.config.get('subsetting', {})
        self.classifier_config = self.config.get('classifier', {})
        self.llm_config = self.config.get('llm_validation', {})
        
    def run(self, 
            seed_data_path: str = None,
            text_column: str = None,
            label_column: str = None) -> Dict[str, Any]:
        """
        Run the complete initial training pipeline.
        
        Args:
            seed_data_path: Path to seed training data
            text_column: Name of text column in data
            label_column: Name of label column in data
            
        Returns:
            Dictionary with training results and statistics
        """
        print("=" * 60)
        print("R.E.D. INITIAL TRAINING PIPELINE")
        print("=" * 60)
        
        start_time = time.time()
        
        # Use provided path or stored path
        if seed_data_path:
            self.seed_data_path = seed_data_path
        
        if not self.seed_data_path:
            raise ValueError("seed_data_path must be provided")
        
        # Set column names
        text_column = text_column or self.config.get('data', {}).get('text_column', 'text')
        label_column = label_column or self.config.get('data', {}).get('label_column', 'label')
        
        try:
            # Step 1: Load seed data
            print("\n" + "="*50)
            print("STEP 1: LOADING SEED DATA")
            print("="*50)
            data_stats = self._load_seed_data(text_column, label_column)
            
            # Step 2: Create label subsets
            print("\n" + "="*50)
            print("STEP 2: CREATING LABEL SUBSETS")
            print("="*50)
            subset_stats = self._create_label_subsets()
            
            # Step 3: Train subset classifiers
            print("\n" + "="*50)
            print("STEP 3: TRAINING SUBSET CLASSIFIERS")
            print("="*50)
            classifier_stats = self._train_subset_classifiers()
            
            # Step 4: Setup LLM validator
            print("\n" + "="*50)
            print("STEP 4: SETTING UP LLM VALIDATOR")
            print("="*50)
            validator_stats = self._setup_llm_validator()
            
            # Step 5: Save all components
            print("\n" + "="*50)
            print("STEP 5: SAVING TRAINED COMPONENTS")
            print("="*50)
            save_stats = self._save_components()
            
            # Compile results
            total_time = time.time() - start_time
            
            results = {
                'status': 'success',
                'total_time': total_time,
                'data_stats': data_stats,
                'subset_stats': subset_stats,
                'classifier_stats': classifier_stats,
                'validator_stats': validator_stats,
                'save_stats': save_stats,
                'output_dir': str(self.output_dir),
                'config': self.config
            }
            
            # Save results summary
            self._save_results_summary(results)
            
            print("\n" + "="*60)
            print("INITIAL TRAINING COMPLETED SUCCESSFULLY")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Output directory: {self.output_dir}")
            print("="*60)
            
            return results
            
        except Exception as e:
            print(f"\nERROR during initial training: {e}")
            error_results = {
                'status': 'error',
                'error': str(e),
                'total_time': time.time() - start_time
            }
            self._save_results_summary(error_results)
            raise
    
    def _load_seed_data(self, text_column: str, label_column: str) -> Dict[str, Any]:
        """Load and validate seed training data."""
        print(f"Loading seed data from: {self.seed_data_path}")
        
        data_stats = self.data_manager.load_seed_data(
            filepath=self.seed_data_path,
            text_column=text_column,
            label_column=label_column
        )
        
        print(f"✓ Loaded {data_stats['total_samples']} samples")
        print(f"✓ Found {data_stats['unique_labels']} unique labels")
        print(f"✓ Average samples per label: {data_stats['avg_samples_per_label']:.1f}")
        
        return data_stats
    
    def _create_label_subsets(self) -> Dict[str, Any]:
        """Create label subsets using greedy selection."""
        print("Creating label subsets using greedy selection...")
        
        # Get unique labels
        unique_labels = list(set(self.data_manager.seed_data['labels']))
        print(f"Working with {len(unique_labels)} unique labels")
        
        # Initialize label subsetter (will use config internally)
        self.label_subsetter = LabelSubsetter()
        
        # Create subsets
        subset_size = self.subsetting_config.get('subset_size', 8)
        self.subset_mapping = self.label_subsetter.create_subsets(
            labels=unique_labels,
            subset_size=subset_size
        )
        
        # Set subset mapping in data manager
        self.data_manager.set_subset_mapping(self.subset_mapping)
        
        # Get statistics
        subset_stats = self.label_subsetter.get_subset_statistics()
        
        print(f"✓ Created {subset_stats['num_subsets']} subsets")
        print(f"✓ Average subset size: {subset_stats['avg_subset_size']:.1f}")
        print(f"✓ Subset size range: {subset_stats['min_subset_size']}-{subset_stats['max_subset_size']}")
        
        return subset_stats
    
    def _train_subset_classifiers(self) -> Dict[str, Any]:
        """Train classifiers for each subset."""
        print("Training subset classifiers...")
        
        classifier_stats = {
            'total_subsets': len(self.subset_mapping),
            'trained_classifiers': 0,
            'failed_classifiers': 0,
            'individual_stats': {}
        }
        
        for subset_id, subset_labels in self.subset_mapping.items():
            print(f"\nTraining classifier for {subset_id} ({len(subset_labels)} labels)")
            
            try:
                # Initialize classifier (will use config internally)
                classifier = SubsetClassifier(
                    subset_id=subset_id,
                    subset_labels=subset_labels
                )
                
                # Get training data for this subset
                subset_data = self.data_manager.get_subset_data(subset_id)
                noise_data = self.data_manager.get_noise_data(subset_id, max_samples=1000)
                
                if len(subset_data['texts']) == 0:
                    print(f"  ⚠ No training data for {subset_id}, skipping...")
                    continue
                
                # Train classifier
                training_stats = classifier.train(
                    texts=subset_data['texts'],
                    labels=subset_data['labels'],
                    noise_texts=noise_data,
                    validation_split=self.classifier_config.get('validation_split', 0.2)
                )
                
                # Store classifier
                self.subset_classifiers[subset_id] = classifier
                classifier_stats['individual_stats'][subset_id] = training_stats
                classifier_stats['trained_classifiers'] += 1
                
                print(f"  ✓ Training accuracy: {training_stats['train_accuracy']:.3f}")
                if 'validation_accuracy' in training_stats:
                    print(f"  ✓ Validation accuracy: {training_stats['validation_accuracy']:.3f}")
                
            except Exception as e:
                print(f"  ✗ Failed to train {subset_id}: {e}")
                classifier_stats['failed_classifiers'] += 1
        
        print(f"\n✓ Successfully trained {classifier_stats['trained_classifiers']} classifiers")
        if classifier_stats['failed_classifiers'] > 0:
            print(f"⚠ Failed to train {classifier_stats['failed_classifiers']} classifiers")
        
        return classifier_stats
    
    def _setup_llm_validator(self) -> Dict[str, Any]:
        """Setup LLM validator with label descriptions."""
        print("Setting up LLM validator...")
        
        # Initialize LLM validator (will use config internally)
        self.llm_validator = LLMValidator()
        
        # Set training data for similarity search
        self.llm_validator.set_training_data(
            training_texts=self.data_manager.seed_data['texts'],
            training_labels=self.data_manager.seed_data['labels']
        )
        
        # Generate label descriptions
        unique_labels = list(set(self.data_manager.seed_data['labels']))
        auto_generate = self.llm_config.get('auto_generate_descriptions', True)
        
        label_descriptions = self.llm_validator.generate_label_descriptions(
            label_names=unique_labels,
            auto_generate=auto_generate
        )
        
        validator_stats = {
            'model_name': self.llm_config.get('model_name'),
            'total_labels': len(unique_labels),
            'descriptions_generated': len(label_descriptions),
            'training_samples': len(self.data_manager.seed_data['texts'])
        }
        
        print(f"✓ Generated descriptions for {len(label_descriptions)} labels")
        print(f"✓ Indexed {len(self.data_manager.seed_data['texts'])} training samples")
        
        return validator_stats
    
    def _save_components(self) -> Dict[str, Any]:
        """Save all trained components to disk."""
        print("Saving trained components...")
        
        # Create component directories
        components_dir = self.output_dir / "components"
        components_dir.mkdir(exist_ok=True)
        
        classifiers_dir = components_dir / "classifiers"
        classifiers_dir.mkdir(exist_ok=True)
        
        save_stats = {
            'saved_components': [],
            'failed_saves': [],
            'output_paths': {}
        }
        
        try:
            # Save label subsetter
            subsetter_path = components_dir / "label_subsetter.pkl"
            self.label_subsetter.save(str(subsetter_path))
            save_stats['saved_components'].append('label_subsetter')
            save_stats['output_paths']['label_subsetter'] = str(subsetter_path)
            
            # Save subset classifiers
            for subset_id, classifier in self.subset_classifiers.items():
                classifier_path = classifiers_dir / f"{subset_id}"
                classifier.save(str(classifier_path))
                save_stats['saved_components'].append(f'classifier_{subset_id}')
                save_stats['output_paths'][f'classifier_{subset_id}'] = str(classifier_path)
            
            # Save LLM validator cache
            validator_cache_path = components_dir / "validator_cache.pkl"
            self.llm_validator.save_cache(str(validator_cache_path))
            save_stats['saved_components'].append('validator_cache')
            save_stats['output_paths']['validator_cache'] = str(validator_cache_path)
            
            # Save data manager state
            data_manager_path = components_dir / "data_manager.pkl"
            self.data_manager.save_state(str(data_manager_path))
            save_stats['saved_components'].append('data_manager')
            save_stats['output_paths']['data_manager'] = str(data_manager_path)
            
            # Save subset mapping
            mapping_path = components_dir / "subset_mapping.json"
            with open(mapping_path, 'w') as f:
                json.dump(self.subset_mapping, f, indent=2)
            save_stats['saved_components'].append('subset_mapping')
            save_stats['output_paths']['subset_mapping'] = str(mapping_path)
            
            print(f"✓ Saved {len(save_stats['saved_components'])} components")
            
        except Exception as e:
            print(f"✗ Error saving components: {e}")
            save_stats['failed_saves'].append(str(e))
        
        return save_stats
    
    def _save_results_summary(self, results: Dict[str, Any]) -> None:
        """Save training results summary."""
        summary_path = self.output_dir / "initial_training_summary.json"
        
        try:
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"✓ Results summary saved to: {summary_path}")
        except Exception as e:
            print(f"⚠ Failed to save results summary: {e}")
    
    def load_trained_components(self, components_dir: str = None) -> None:
        """
        Load previously trained components.
        
        Args:
            components_dir: Directory containing saved components
        """
        if components_dir is None:
            components_dir = self.output_dir / "components"
        else:
            components_dir = Path(components_dir)
        
        print(f"Loading trained components from: {components_dir}")
        
        try:
            # Load label subsetter
            subsetter_path = components_dir / "label_subsetter.pkl"
            if subsetter_path.exists():
                self.label_subsetter = LabelSubsetter.load(str(subsetter_path))
                print("✓ Loaded label subsetter")
            
            # Load subset mapping
            mapping_path = components_dir / "subset_mapping.json"
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    self.subset_mapping = json.load(f)
                self.data_manager.set_subset_mapping(self.subset_mapping)
                print("✓ Loaded subset mapping")
            
            # Load subset classifiers
            classifiers_dir = components_dir / "classifiers"
            if classifiers_dir.exists():
                for subset_id in self.subset_mapping.keys():
                    classifier_path = classifiers_dir / f"{subset_id}"
                    if classifier_path.exists():
                        self.subset_classifiers[subset_id] = SubsetClassifier.load(str(classifier_path))
                        print(f"✓ Loaded classifier for {subset_id}")
            
            # Load data manager state
            data_manager_path = components_dir / "data_manager.pkl"
            if data_manager_path.exists():
                self.data_manager.load_state(str(data_manager_path))
                print("✓ Loaded data manager state")
            
            # Initialize LLM validator
            self.llm_validator = LLMValidator(
                model_name=self.llm_config.get('model_name', 'deepseek-r1-8b'),
                embedding_model=self.config.get('embeddings', {}).get('model_name', 'all-MiniLM-L6-v2'),
                temperature=self.llm_config.get('temperature', 0.0),
                max_timeout=self.llm_config.get('max_timeout', 30),
                similar_examples_count=self.llm_config.get('similar_examples_count', 3)
            )
            
            # Load validator cache
            validator_cache_path = components_dir / "validator_cache.pkl"
            if validator_cache_path.exists():
                self.llm_validator.load_cache(str(validator_cache_path))
                print("✓ Loaded validator cache")
            
            print("✓ All components loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading components: {e}")
            raise
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the current training state."""
        summary = {
            'subset_mapping': self.subset_mapping,
            'num_subsets': len(self.subset_mapping),
            'trained_classifiers': len(self.subset_classifiers),
            'data_stats': self.data_manager.get_data_statistics(),
            'output_dir': str(self.output_dir)
        }
        
        if self.llm_validator:
            summary['validator_stats'] = self.llm_validator.get_validation_statistics()
        
        return summary

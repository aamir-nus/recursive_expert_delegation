"""
Entry point script for R.E.D. initial training.

This script sets up and runs the initial training pipeline for the R.E.D. framework,
including subset creation and initial classifier training.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from red.pipelines.initial_training import InitialTrainingPipeline
from red.config.config_loader import get_config_loader

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run R.E.D. initial training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with CSV file
  python run_initial_training.py --seed-data data/train.csv

  # Specify custom columns and output directory
  python run_initial_training.py \\
    --seed-data data/train.csv \\
    --text-column "review_text" \\
    --label-column "category" \\
    --output-dir ./my_outputs

  # Use custom config directory
  python run_initial_training.py \\
    --seed-data data/train.csv \\
    --config-dir ./my_config
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--seed-data', 
        type=str, 
        required=True,
        help='Path to seed training data file (CSV, JSON, or pickle)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=None,
        help='Directory to save outputs (default: from config)'
    )
    
    parser.add_argument(
        '--config-dir', 
        type=str, 
        default=None,
        help='Directory containing configuration files (default: built-in config)'
    )
    
    parser.add_argument(
        '--text-column', 
        type=str, 
        default=None,
        help='Name of text column in data (default: from config)'
    )
    
    parser.add_argument(
        '--label-column', 
        type=str, 
        default=None,
        help='Name of label column in data (default: from config)'
    )
    
    # Training configuration overrides
    parser.add_argument(
        '--subset-size', 
        type=int, 
        default=None,
        help='Maximum number of labels per subset'
    )
    
    parser.add_argument(
        '--llm-model', 
        type=str, 
        default=None,
        help='LLM model to use for validation'
    )
    
    parser.add_argument(
        '--embedding-model', 
        type=str, 
        default=None,
        help='Embedding model to use'
    )
    
    parser.add_argument(
        '--classifier-type', 
        type=str, 
        choices=['logistic_regression', 'random_forest'],
        default=None,
        help='Type of classifier to train'
    )
    
    # Utility flags
    parser.add_argument(
        '--verbose', 
        '-v', 
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show configuration and exit without training'
    )
    
    parser.add_argument(
        '--save-config', 
        type=str, 
        default=None,
        help='Save effective configuration to file'
    )
    
    return parser.parse_args()

def update_config_from_args(config, args):
    """Update configuration with command line arguments."""
    if args.subset_size is not None:
        config.setdefault('subsetting', {})['subset_size'] = args.subset_size
    
    if args.llm_model is not None:
        config.setdefault('llm_validation', {})['model_name'] = args.llm_model
    
    if args.embedding_model is not None:
        config.setdefault('embeddings', {})['model_name'] = args.embedding_model
    
    if args.classifier_type is not None:
        config.setdefault('classifier', {})['type'] = args.classifier_type
    
    return config

def validate_inputs(args):
    """Validate input arguments."""
    # Check if seed data file exists
    if not os.path.exists(args.seed_data):
        raise FileNotFoundError(f"Seed data file not found: {args.seed_data}")
    
    # Check file extension
    valid_extensions = ['.csv', '.json', '.pkl', '.pickle']
    file_ext = Path(args.seed_data).suffix.lower()
    if file_ext not in valid_extensions:
        raise ValueError(f"Unsupported file format: {file_ext}. "
                        f"Supported formats: {valid_extensions}")
    
    # Check config directory if provided
    if args.config_dir and not os.path.exists(args.config_dir):
        raise FileNotFoundError(f"Config directory not found: {args.config_dir}")

def print_configuration(config, args):
    """Print the effective configuration."""
    print("=" * 60)
    print("R.E.D. INITIAL TRAINING CONFIGURATION")
    print("=" * 60)
    
    print(f"Seed data file: {args.seed_data}")
    print(f"Text column: {args.text_column or config.get('data', {}).get('text_column', 'text')}")
    print(f"Label column: {args.label_column or config.get('data', {}).get('label_column', 'label')}")
    print(f"Output directory: {args.output_dir or config.get('data', {}).get('output_dir', './outputs')}")
    
    print(f"\nSubsetting Configuration:")
    subsetting = config.get('subsetting', {})
    print(f"  Subset size: {subsetting.get('subset_size', 8)}")
    print(f"  Use UMAP: {subsetting.get('use_umap', True)}")
    print(f"  UMAP components: {subsetting.get('umap_components', 50)}")
    
    print(f"\nClassifier Configuration:")
    classifier = config.get('classifier', {})
    print(f"  Type: {classifier.get('type', 'logistic_regression')}")
    print(f"  Use embeddings: {classifier.get('use_embeddings', True)}")
    print(f"  Noise oversample factor: {classifier.get('noise_oversample_factor', 2.0)}")
    
    print(f"\nLLM Validation Configuration:")
    llm = config.get('llm_validation', {})
    print(f"  Model: {llm.get('model_name', 'deepseek-r1-8b')}")
    print(f"  Temperature: {llm.get('temperature', 0.0)}")
    print(f"  Auto-generate descriptions: {llm.get('auto_generate_descriptions', True)}")
    
    print(f"\nEmbedding Configuration:")
    embeddings = config.get('embeddings', {})
    print(f"  Model: {embeddings.get('model_name', 'all-MiniLM-L6-v2')}")
    print(f"  Enable cache: {embeddings.get('enable_cache', True)}")
    
    print("=" * 60)

def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate inputs
        validate_inputs(args)
        
        # Load and update configuration
        config_loader = get_config_loader(args.config_dir)
        config = config_loader.load_main_config()
        config = update_config_from_args(config, args)
        
        # Print configuration
        if args.verbose or args.dry_run:
            print_configuration(config, args)
        
        # Save configuration if requested
        if args.save_config:
            with open(args.save_config, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Configuration saved to: {args.save_config}")
        
        # Exit if dry run
        if args.dry_run:
            print("Dry run completed. Use --verbose to see full configuration.")
            return
        
        # Initialize and run pipeline
        pipeline = InitialTrainingPipeline(
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            seed_data_path=args.seed_data
        )
        
        # Run initial training
        results = pipeline.run(
            seed_data_path=args.seed_data,
            text_column=args.text_column,
            label_column=args.label_column
        )
        
        # Print summary
        if results['status'] == 'success':
            print(f"\n✓ Initial training completed successfully!")
            print(f"  - Trained {results['classifier_stats']['trained_classifiers']} classifiers")
            print(f"  - Created {results['subset_stats']['num_subsets']} subsets")
            print(f"  - Total time: {results['total_time']:.2f} seconds")
            print(f"  - Output saved to: {results['output_dir']}")
            
            # Show next steps
            print(f"\nNext steps:")
            print(f"1. Prepare your unlabeled data file")
            print(f"2. Run active learning:")
            print(f"   python run_active_learning.py \\")
            print(f"     --components-dir {results['output_dir']}/components \\")
            print(f"     --unlabeled-data your_unlabeled_data.csv")
        else:
            print(f"\n✗ Initial training failed: {results.get('error', 'Unknown error')}")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        return 1
    
    except Exception as e:
        print(f"\nERROR: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

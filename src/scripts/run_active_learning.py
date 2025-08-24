"""
Entry point script for R.E.D. active learning.

This script runs the main active learning loop of the R.E.D. framework,
processing unlabeled data and iteratively improving classifiers.
"""

import sys
import argparse
import os
from pathlib import Path
import json

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from red.pipelines.active_learning import ActiveLearningLoop
from red.config.config_loader import get_config_loader

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run R.E.D. active learning loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_active_learning.py \\
    --components-dir ./outputs/components \\
    --unlabeled-data data/unlabeled.csv

  # Custom parameters
  python run_active_learning.py \\
    --components-dir ./outputs/components \\
    --unlabeled-data data/unlabeled.csv \\
    --max-iterations 20 \\
    --batch-size 200 \\
    --samples-per-iteration 100

  # Resume from checkpoint
  python run_active_learning.py \\
    --components-dir ./outputs/components \\
    --unlabeled-data data/unlabeled.csv \\
    --resume-from ./outputs/checkpoint_iter_5
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--components-dir', 
        type=str, 
        required=True,
        help='Directory containing trained components from initial training'
    )
    
    parser.add_argument(
        '--unlabeled-data', 
        type=str, 
        required=True,
        help='Path to unlabeled data file (CSV, JSON, or text)'
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
        help='Name of text column in unlabeled data (default: from config)'
    )
    
    # Active learning parameters
    parser.add_argument(
        '--max-iterations', 
        type=int, 
        default=None,
        help='Maximum number of active learning iterations'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=None,
        help='Size of data batches to process'
    )
    
    parser.add_argument(
        '--samples-per-iteration', 
        type=int, 
        default=None,
        help='Number of samples to validate per iteration'
    )
    
    parser.add_argument(
        '--retrain-threshold', 
        type=int, 
        default=None,
        help='Number of validated samples before retraining'
    )
    
    # LLM configuration overrides
    parser.add_argument(
        '--llm-model', 
        type=str, 
        default=None,
        help='LLM model to use for validation'
    )
    
    parser.add_argument(
        '--confidence-threshold', 
        type=float, 
        default=None,
        help='Minimum confidence threshold for accepting validations'
    )
    
    parser.add_argument(
        '--disable-cache', 
        action='store_true',
        help='Disable LLM validation cache'
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
        help='Show configuration and exit without running'
    )
    
    parser.add_argument(
        '--resume-from', 
        type=str, 
        default=None,
        help='Resume from checkpoint directory'
    )
    
    parser.add_argument(
        '--save-checkpoints', 
        action='store_true',
        help='Save checkpoints during training'
    )
    
    parser.add_argument(
        '--checkpoint-frequency', 
        type=int, 
        default=5,
        help='Save checkpoint every N iterations'
    )
    
    return parser.parse_args()

def update_config_from_args(config, args):
    """Update configuration with command line arguments."""
    if args.max_iterations is not None:
        config.setdefault('active_learning', {})['max_iterations'] = args.max_iterations
    
    if args.batch_size is not None:
        config.setdefault('active_learning', {})['batch_size'] = args.batch_size
    
    if args.samples_per_iteration is not None:
        config.setdefault('active_learning', {})['samples_per_iteration'] = args.samples_per_iteration
    
    if args.retrain_threshold is not None:
        config.setdefault('active_learning', {})['retrain_threshold'] = args.retrain_threshold
    
    if args.llm_model is not None:
        config.setdefault('llm_validation', {})['model_name'] = args.llm_model
    
    if args.confidence_threshold is not None:
        config.setdefault('llm_validation', {})['confidence_threshold'] = args.confidence_threshold
    
    if args.disable_cache:
        config.setdefault('llm_validation', {})['use_cache'] = False
    
    return config

def validate_inputs(args):
    """Validate input arguments."""
    # Check if components directory exists
    if not os.path.exists(args.components_dir):
        raise FileNotFoundError(f"Components directory not found: {args.components_dir}")
    
    # Check if required component files exist
    required_files = [
        'subset_mapping.json',
        'data_manager.pkl',
        'classifiers'
    ]
    
    for required_file in required_files:
        file_path = Path(args.components_dir) / required_file
        if not file_path.exists():
            raise FileNotFoundError(f"Required component not found: {file_path}")
    
    # Check if unlabeled data file exists
    if not os.path.exists(args.unlabeled_data):
        raise FileNotFoundError(f"Unlabeled data file not found: {args.unlabeled_data}")
    
    # Check file extension
    valid_extensions = ['.csv', '.json', '.pkl', '.pickle', '.txt']
    file_ext = Path(args.unlabeled_data).suffix.lower()
    if file_ext not in valid_extensions:
        raise ValueError(f"Unsupported file format: {file_ext}. "
                        f"Supported formats: {valid_extensions}")
    
    # Check config directory if provided
    if args.config_dir and not os.path.exists(args.config_dir):
        raise FileNotFoundError(f"Config directory not found: {args.config_dir}")
    
    # Check resume directory if provided
    if args.resume_from and not os.path.exists(args.resume_from):
        raise FileNotFoundError(f"Resume directory not found: {args.resume_from}")

def print_configuration(config, args):
    """Print the effective configuration."""
    print("=" * 60)
    print("R.E.D. ACTIVE LEARNING CONFIGURATION")
    print("=" * 60)
    
    print(f"Components directory: {args.components_dir}")
    print(f"Unlabeled data file: {args.unlabeled_data}")
    print(f"Text column: {args.text_column or config.get('data', {}).get('text_column', 'text')}")
    print(f"Output directory: {args.output_dir or config.get('data', {}).get('output_dir', './outputs')}")
    
    print(f"\nActive Learning Configuration:")
    al_config = config.get('active_learning', {})
    print(f"  Max iterations: {al_config.get('max_iterations', 10)}")
    print(f"  Batch size: {al_config.get('batch_size', 100)}")
    print(f"  Samples per iteration: {al_config.get('samples_per_iteration', 50)}")
    print(f"  Retrain threshold: {al_config.get('retrain_threshold', 100)}")
    print(f"  Min information gain: {al_config.get('min_information_gain', 0.01)}")
    
    print(f"\nLLM Validation Configuration:")
    llm_config = config.get('llm_validation', {})
    print(f"  Model: {llm_config.get('model_name', 'deepseek-r1-8b')}")
    print(f"  Temperature: {llm_config.get('temperature', 0.0)}")
    print(f"  Confidence threshold: {llm_config.get('confidence_threshold', 0.5)}")
    print(f"  Use cache: {llm_config.get('use_cache', True)}")
    print(f"  Similar examples count: {llm_config.get('similar_examples_count', 3)}")
    
    if args.resume_from:
        print(f"\nResume Configuration:")
        print(f"  Resume from: {args.resume_from}")
    
    if args.save_checkpoints:
        print(f"\nCheckpoint Configuration:")
        print(f"  Save checkpoints: {args.save_checkpoints}")
        print(f"  Checkpoint frequency: {args.checkpoint_frequency}")
    
    print("=" * 60)

def load_components_info(components_dir):
    """Load information about the trained components."""
    info = {}
    
    # Load subset mapping
    mapping_path = Path(components_dir) / "subset_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            subset_mapping = json.load(f)
        info['num_subsets'] = len(subset_mapping)
        info['total_labels'] = sum(len(labels) for labels in subset_mapping.values())
    
    # Count classifiers
    classifiers_dir = Path(components_dir) / "classifiers"
    if classifiers_dir.exists():
        classifier_files = list(classifiers_dir.glob("*_metadata.pkl"))
        info['num_classifiers'] = len(classifier_files)
    
    return info

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
            
            # Show component information
            components_info = load_components_info(args.components_dir)
            if components_info:
                print(f"\nLoaded Components:")
                print(f"  Subsets: {components_info.get('num_subsets', 'Unknown')}")
                print(f"  Labels: {components_info.get('total_labels', 'Unknown')}")
                print(f"  Classifiers: {components_info.get('num_classifiers', 'Unknown')}")
                print("=" * 60)
        
        # Exit if dry run
        if args.dry_run:
            print("Dry run completed. Use --verbose to see full configuration.")
            return
        
        # Initialize active learning loop
        loop = ActiveLearningLoop(
            components_dir=args.components_dir,
            output_dir=args.output_dir,
            config_dir=args.config_dir
        )
        
        # Run active learning
        results = loop.run(
            unlabeled_data_path=args.unlabeled_data,
            max_iterations=args.max_iterations,
            batch_size=args.batch_size,
            samples_per_iteration=args.samples_per_iteration
        )
        
        # Print summary
        if results['status'] == 'success':
            print(f"\n✓ Active learning completed successfully!")
            print(f"  - Iterations: {results['total_iterations']}")
            print(f"  - Validated samples: {results['total_validated_samples']}")
            print(f"  - Convergence: {'Yes' if results['convergence_achieved'] else 'No'}")
            print(f"  - Total time: {results['total_time']:.2f} seconds")
            print(f"  - Output saved to: {results['output_dir']}")
            
            # Show performance stats
            perf_stats = results.get('performance_stats', {})
            if perf_stats:
                print(f"\nPerformance:")
                print(f"  - Avg iteration time: {perf_stats.get('average_iteration_time', 0):.2f}s")
                print(f"  - Validation efficiency: {perf_stats.get('validation_efficiency', 0):.2f} samples/sec")
            
            # Show final validation stats
            if results['validation_history']:
                final_validation = results['validation_history'][-1]
                print(f"\nFinal Validation Rate: {final_validation['validation_rate']:.1%}")
            
            print(f"\nNext steps:")
            print(f"1. Review results in: {results['output_dir']}")
            print(f"2. Use final trained classifiers in: {results['output_dir']}/final_state")
            print(f"3. Export training data from: {results['output_dir']}/final_state/final_training_data.csv")
        else:
            print(f"\n✗ Active learning failed: {results.get('error', 'Unknown error')}")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nActive learning interrupted by user.")
        return 1
    
    except Exception as e:
        print(f"\nERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

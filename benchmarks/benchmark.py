#!/usr/bin/env python3
"""
R.E.D. Framework Benchmark Script

This script runs comprehensive benchmarks comparing different training data sizes
for the Recursive Expert Delegation framework.

LOGGING ARCHITECTURE:
├── Main Benchmark Logger (main_benchmark_YYYYMMDD_HHMMSS.log)
│   ├── Overall benchmark progress and timing
│   ├── Dataset statistics and class distribution
│   ├── Experiment status updates
│   ├── Comparison table output
│   └── File save confirmations
│
├── Individual Experiment Logs
│   ├── run_30_samples_YYYYMMDD_HHMMSS.log
│   ├── run_50_samples_YYYYMMDD_HHMMSS.log
│   └── run_100_samples_YYYYMMDD_HHMMSS.log
│       ├── Data preparation steps
│       ├── Initial training progress
│       ├── Active learning iterations
│       ├── Detailed evaluation metrics:
│       │   ├── Overall accuracy, precision, recall, F1-score
│       │   ├── Per-class performance metrics
│       │   └── Confusion matrix
│       ├── Error handling and warnings
│       └── Performance statistics
│
├── Output Files
│   ├── benchmark_comparison_YYYYMMDD_HHMMSS.csv (summary table)
│   ├── detailed_results_YYYYMMDD_HHMMSS.json (complete structured results)
│   └── Individual experiment directories with models and data
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from red.core.classifier import SubsetClassifier
from red.pipelines.initial_training import InitialTrainingPipeline
from red.pipelines.active_learning import ActiveLearningLoop

def sanitize_subset_id(subset_id):
    """Sanitize subset_id for safe file paths."""
    return str(subset_id).replace(' ', '_').replace('/', '_')

def setup_logging(log_dir: Path, run_name: str):
    """Setup logging for a specific run."""
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{run_name}_{timestamp}.log"
    
    # Configure logger
    logger = logging.getLogger(f"benchmark_{run_name}")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger, log_file

def split_dataframe(dataframe, samples_per_class: int = 100, logger=None):
    """
    Split the dataframe into training and prediction data.
    Each class will have samples_per_class samples for training and the rest for prediction.

    Args:
        dataframe: the dataframe to split
        samples_per_class: the number of samples to use for training per class
        logger: logger instance for logging information

    Returns:
        train_data: the dataframe for training
        prediction_data: the dataframe for prediction
    """

    # First, filter out classes that don't have enough samples
    class_counts = dataframe.groupby('label_name').size()

    # Find classes that have at least samples_per_class + 10 samples (to leave some for testing)
    min_samples_needed = samples_per_class + 0
    valid_classes = class_counts[class_counts >= min_samples_needed].index.tolist()

    if logger:
        logger.info(f"Original dataset: {len(class_counts)} classes")
        logger.info(f"Classes with >= {min_samples_needed} samples: {len(valid_classes)}")
        if len(valid_classes) == 0:
            logger.warning(f"No classes have enough samples (>= {min_samples_needed}). Using classes with most samples available.")

    if len(valid_classes) == 0:
        # If no class has enough samples, use the classes with the most samples
        valid_classes = class_counts.nlargest(min(50, len(class_counts))).index.tolist()
        if logger:
            logger.info(f"Using top {len(valid_classes)} classes by sample count")

    # Filter dataframe to only include valid classes
    filtered_df = dataframe[dataframe['label_name'].isin(valid_classes)]

    if logger:
        logger.info(f"Filtered dataset: {len(filtered_df)} samples from {len(valid_classes)} classes")

    # Now sample exactly samples_per_class from each valid class
    train_data_list = []

    for class_name in valid_classes:
        class_data = filtered_df[filtered_df['label_name'] == class_name]

        # Sample exactly samples_per_class (or all available if less)
        n_samples = min(samples_per_class, len(class_data))
        sampled_class_data = class_data.sample(n=n_samples, random_state=42)
        train_data_list.append(sampled_class_data)

        if logger:
            logger.info(f"  {class_name}: {n_samples} samples (out of {len(class_data)} available)")

    # Combine all training data
    train_data = pd.concat(train_data_list, ignore_index=True)

    # Get remaining data for testing by finding samples not in training data
    train_indices = set()
    for sampled_class_data in train_data_list:
        train_indices.update(sampled_class_data.index)

    prediction_data = filtered_df[~filtered_df.index.isin(train_indices)]

    if logger:
        logger.info(f"Training data: {len(train_data)} samples")
        logger.info(f"Test data: {len(prediction_data)} samples")
    
    return train_data, prediction_data

def save_data(train_texts, train_labels, unlabeled_texts, data_dir, logger):
    """Save the generated data to files."""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    # Save training data
    train_df = pd.DataFrame({
        'text': train_texts,
        'label': train_labels
    })
    train_path = data_dir / "train.csv"
    train_df.to_csv(train_path, index=False)
    
    # Save unlabeled data
    unlabeled_df = pd.DataFrame({
        'text': unlabeled_texts
    })
    unlabeled_path = data_dir / "unlabeled.csv"
    unlabeled_df.to_csv(unlabeled_path, index=False)
    
    logger.info(f"Training data saved: {train_path} ({len(train_texts)} samples)")
    logger.info(f"Unlabeled data saved: {unlabeled_path} ({len(unlabeled_texts)} samples)")
    
    return str(train_path), str(unlabeled_path)

def evaluate_predictions(true_labels, predicted_labels, logger):
    """Evaluate model predictions and return comprehensive metrics."""

    # Calculate basic metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, average='weighted', zero_division=0
    )

    # Get per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Get unique labels
    unique_labels = sorted(list(set(true_labels) | set(predicted_labels)))

    # Log results (also print to console for immediate feedback)
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")

    logger.info(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Precision: {precision:.4f}")

    logger.info(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted Recall: {recall:.4f}")

    logger.info(f"Weighted F1-Score: {f1:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")

    logger.info("\nPer-Class Metrics:")
    print("\nPer-Class Metrics:")
    for i, label in enumerate(unique_labels):
        if i < len(precision_per_class):
            class_metrics = f"  {label}: P={precision_per_class[i]:.4f}, R={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}, Support={support_per_class[i]}"
            logger.info(class_metrics)
            print(class_metrics)

    cm_str = f"\nConfusion Matrix:\n{cm}"
    logger.info(cm_str)
    print(cm_str)

    # Return structured results
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {
            'labels': unique_labels,
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1_score': f1_per_class.tolist(),
            'support': support_per_class.tolist()
        }
    }

def load_final_training_data(output_dir):
    """Load the final training data after active learning."""
    try:
        final_data_path = Path(output_dir) / "final_state" / "final_training_data.csv"
        if final_data_path.exists():
            df = pd.read_csv(final_data_path)
            return df['text'].tolist(), df['label'].tolist()
    except Exception:
        pass
    return [], []

def predict_on_test_data(components_dir, test_texts, logger, test_labels=None):
    """Make predictions on test data using trained classifiers and subset mapping."""
    try:
        # Load subset mapping (subset_id -> [labels])
        subset_mapping_path = os.path.join(components_dir, "subset_mapping.json")
        if not os.path.exists(subset_mapping_path):
            logger.error(f"Subset mapping not found: {subset_mapping_path}")
            return []
        with open(subset_mapping_path, 'r') as f:
            subset_mapping = json.load(f)
        
        # Create reverse mapping (label -> subset_id)
        label_to_subset = {}
        for subset_id, labels in subset_mapping.items():
            for label in labels:
                label_to_subset[label] = subset_id
        
        # Load classifiers by subset_id
        classifiers_dir = os.path.join(components_dir, "classifiers")
        subset_ids = set(subset_mapping.keys())
        classifiers = {}
        for subset_id in subset_ids:
            safe_id = sanitize_subset_id(subset_id)
            classifier_path = os.path.join(classifiers_dir, f"{safe_id}")
            if os.path.exists(f"{classifier_path}_metadata.pkl"):
                classifiers[subset_id] = SubsetClassifier.load(classifier_path)
            else:
                logger.warning(f"Classifier for subset {subset_id} not found at {classifier_path}")
        
        predictions = []
        if test_labels is not None:
            for text, label in zip(test_texts, test_labels):
                subset_id = label_to_subset.get(label)
                if subset_id is None:
                    logger.warning(f"Label '{label}' not found in subset mapping")
                    pred = SubsetClassifier.NOISE_LABEL
                else:
                    clf = classifiers.get(subset_id)
                    if clf:
                        try:
                            pred = clf.predict([text])[0]
                        except Exception as e:
                            logger.warning(f"Prediction failed for subset {subset_id}: {e}")
                            pred = SubsetClassifier.NOISE_LABEL
                    else:
                        logger.warning(f"No classifier found for subset_id {subset_id} (label: {label})")
                        pred = SubsetClassifier.NOISE_LABEL
                predictions.append(pred)
        else:
            logger.error("test_labels must be provided for correct subset mapping.")
            return []
        
        logger.info(f"Generated {len(predictions)} predictions using {len(classifiers)} classifiers")
        logger.info(f"Label to subset mapping contains {len(label_to_subset)} labels")
        return predictions
    except Exception as e:
        error_msg = f"Prediction failed: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        print(error_msg)
        return []

def run_single_experiment(samples_per_class, run_id, base_dir, dataframe, test_data):
    """Run a single experiment with specified samples per class."""

    run_name = f"run_{samples_per_class}_samples"
    run_dir = base_dir / run_name
    log_dir = run_dir / "logs"
    data_dir = run_dir / "data"
    output_dir = run_dir / "outputs"

    # Create directories
    for directory in [run_dir, log_dir, data_dir, output_dir]:
        directory.mkdir(exist_ok=True, parents=True)

    # Setup logging
    logger, log_file = setup_logging(log_dir, run_name)

    logger.info("=" * 60)
    logger.info(f"R.E.D. FRAMEWORK BENCHMARK - {samples_per_class} SAMPLES PER CLASS")
    logger.info("=" * 60)

    try:
        # Step 1: Prepare training data
        logger.info("1. PREPARING TRAINING DATA")
        logger.info("-" * 30)

        # Split data: use samples_per_class for training, rest for active learning pool
        train_data, active_learning_pool = split_dataframe(dataframe, samples_per_class, logger)

        train_texts = train_data['text'].tolist()
        train_labels = train_data['label_name'].tolist()
        unlabeled_texts = active_learning_pool['text'].tolist()
        
        train_path, unlabeled_path = save_data(train_texts, train_labels, unlabeled_texts, data_dir, logger)
        
        logger.info(f"Created {len(set(train_labels))} categories:")
        label_counts = pd.Series(train_labels).value_counts()
        for label, count in label_counts.items():
            logger.info(f"  - {label}: {count} samples")
        
        # Step 2: Initial Training
        logger.info("2. INITIAL TRAINING")
        logger.info("-" * 30)
        
        pipeline = InitialTrainingPipeline(output_dir=str(output_dir))
        
        initial_results = pipeline.run(
            seed_data_path=train_path,
            text_column='text',
            label_column='label'
        )
        
        if initial_results['status'] != 'success':
            logger.error(f"Initial training failed: {initial_results.get('error')}")
            return None
        
        logger.info("Initial training completed:")
        logger.info(f"  - Created {initial_results['subset_stats']['num_subsets']} subsets")
        logger.info(f"  - Trained {initial_results['classifier_stats']['trained_classifiers']} classifiers")
        logger.info(f"  - Time: {initial_results['total_time']:.2f} seconds")
        
        # Step 3: Active Learning
        logger.info("3. ACTIVE LEARNING")
        logger.info("-" * 30)
        
        components_dir = output_dir / "components"
        
        loop = ActiveLearningLoop(
            components_dir=str(components_dir),
            output_dir=str(output_dir)
        )
        
        al_results = loop.run(
            unlabeled_data_path=unlabeled_path,
            max_iterations=10,
            batch_size=20,
            samples_per_iteration=100
        )
        
        if al_results['status'] != 'success':
            logger.error(f"Active learning failed: {al_results.get('error')}")
            return None
        
        logger.info("Active learning completed:")
        logger.info(f"  - Iterations: {al_results['total_iterations']}")
        logger.info(f"  - Validated samples: {al_results['total_validated_samples']}")
        logger.info(f"  - Convergence: {'Yes' if al_results['convergence_achieved'] else 'No'}")
        logger.info(f"  - Time: {al_results['total_time']:.2f} seconds")
        
        # Step 4: Evaluation on test data
        logger.info("4. EVALUATION ON TEST DATA")
        logger.info("-" * 30)
        
        test_texts = test_data['text'].tolist()
        true_labels = test_data['label_name'].tolist()
        
        # Make predictions (simplified for demo)
        predicted_labels = predict_on_test_data(components_dir, test_texts, logger, true_labels)
        
        if predicted_labels:
            # Ensure we have the same number of predictions as test samples
            min_len = min(len(true_labels), len(predicted_labels))
            true_labels = true_labels[:min_len]
            predicted_labels = predicted_labels[:min_len]
            
            evaluation_results = evaluate_predictions(true_labels, predicted_labels, logger)
        else:
            logger.warning("No predictions available for evaluation")
            evaluation_results = None
        
        # Compile experiment results
        experiment_results = {
            'samples_per_class': samples_per_class,
            'run_name': run_name,
            'initial_training_time': initial_results['total_time'],
            'active_learning_time': al_results['total_time'],
            'total_time': initial_results['total_time'] + al_results['total_time'],
            'total_iterations': al_results['total_iterations'],
            'total_validated_samples': al_results['total_validated_samples'],
            'convergence_achieved': al_results['convergence_achieved'],
            'evaluation': evaluation_results,
            'log_file': str(log_file),
            'output_dir': str(output_dir)
        }
        
        logger.info("=" * 60)
        logger.info(f"EXPERIMENT COMPLETED SUCCESSFULLY - {samples_per_class} samples")
        logger.info("=" * 60)
        
        return experiment_results
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        logger.error(traceback.format_exc())
        return None

def create_comparison_table(all_results, output_file, logger=None):
    """Create a comparison table of all experiment results."""

    comparison_data = []

    for result in all_results:
        if result and result.get('evaluation'):
            eval_data = result['evaluation']
            comparison_data.append({
                'Samples_Per_Class': result['samples_per_class'],
                'Total_Time_s': result['total_time'],
                'AL_Iterations': result['total_iterations'],
                'Validated_Samples': result['total_validated_samples'],
                'Convergence': result['convergence_achieved'],
                'Accuracy': eval_data['accuracy'],
                'Precision': eval_data['precision'],
                'Recall': eval_data['recall'],
                'F1_Score': eval_data['f1_score']
            })

    # Create DataFrame and save
    df = pd.DataFrame(comparison_data)
    df.to_csv(output_file, index=False)

    # Create log-friendly output
    table_str = df.to_string(index=False, float_format='%.4f')

    comparison_output = "\n" + "=" * 80 + "\nBENCHMARK COMPARISON TABLE\n" + "=" * 80 + "\n" + table_str + "\n" + "=" * 80

    if logger:
        logger.info(comparison_output)

    print(comparison_output)

    return df

def main():
    """Run the R.E.D. framework benchmark with multiple training sizes."""

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_id = f"benchmark_{timestamp}"
    base_dir = Path(__file__).parent / benchmark_id
    base_dir.mkdir(exist_ok=True)

    # Create a main benchmark logger
    main_log_dir = base_dir / "main_logs"
    main_log_dir.mkdir(exist_ok=True)
    main_logger, main_log_file = setup_logging(main_log_dir, "main_benchmark")

    # Log header information
    header_msg = "=" * 80 + "\nR.E.D. FRAMEWORK - COMPREHENSIVE BENCHMARK\n" + "=" * 80
    main_logger.info(header_msg)
    print(header_msg)

    benchmark_info = f"Benchmark ID: {benchmark_id}\nOutput directory: {base_dir}\nMain log file: {main_log_file}"
    main_logger.info(benchmark_info)
    print(benchmark_info)

    try:
        # Load and prepare data
        load_msg = "\nLoading dataset..."
        main_logger.info(load_msg)
        print(load_msg)

        dataframe = pd.read_excel("Datasets/short-text-classification.xlsx")

        # Limit to 5,000 samples and shuffle
        dataframe = dataframe.sample(n=5000, random_state=42).reset_index(drop=True)

        dataset_info = f"Total samples: {len(dataframe)}\nClasses: {dataframe['label_name'].nunique()}"
        main_logger.info(dataset_info)
        print(dataset_info)

        # Show class distribution
        class_counts = dataframe['label_name'].value_counts()
        main_logger.info("\nClass distribution (showing top 20 classes):")
        print("\nClass distribution (showing top 20 classes):")

        # Show top 20 classes and their sample counts
        top_classes = class_counts.head(20)
        for label, count in top_classes.items():
            class_info = f"  {label}: {count} samples"
            main_logger.info(class_info)
            print(class_info)

        if len(class_counts) > 20:
            remaining_info = f"  ... and {len(class_counts) - 20} more classes"
            main_logger.info(remaining_info)
            print(remaining_info)

        # Show statistics about class sizes
        class_stats = f"\nClass size statistics:\n  Min samples per class: {class_counts.min()}\n  Max samples per class: {class_counts.max()}\n  Mean samples per class: {class_counts.mean():.1f}\n  Median samples per class: {class_counts.median()}"
        main_logger.info(class_stats)
        print(class_stats)

        # Define experiment configurations
        # samples_per_class_configs = [30, 50, 100]
        # samples_per_class_configs = [100]
        samples_per_class_configs = [100, 50, 30]
        config_msg = f"\nRunning {len(samples_per_class_configs)} experiments with samples per class: {samples_per_class_configs}"
        main_logger.info(config_msg)
        print(config_msg)

        # Run experiments
        all_results = []

        for i, samples_per_class in enumerate(samples_per_class_configs, 1):
            exp_header = f"\n{'='*20} EXPERIMENT {i}/3: {samples_per_class} SAMPLES PER CLASS {'='*20}"
            main_logger.info(exp_header)
            print(exp_header)

                        # Create train/test split for this specific experiment
            train_data, test_data = split_dataframe(dataframe, samples_per_class, main_logger)

            result = run_single_experiment(
                samples_per_class=samples_per_class,
                run_id=f"exp_{i}",
                base_dir=base_dir,
                dataframe=dataframe,  # Pass full dataframe so it can split properly
                test_data=test_data
            )

            all_results.append(result)

            if result:
                success_msg = f"✓ Experiment {i} completed successfully"
                main_logger.info(success_msg)
                print(success_msg)
            else:
                failure_msg = f"✗ Experiment {i} failed"
                main_logger.error(failure_msg)
                print(failure_msg)

        # Create comparison table
        comparison_file = base_dir / f"benchmark_comparison_{timestamp}.csv"
        main_logger.info(f"\nCreating comparison table: {comparison_file}")
        print(f"\nCreating comparison table: {comparison_file}")
        create_comparison_table(all_results, comparison_file, main_logger)

        # Save detailed results
        results_file = base_dir / f"detailed_results_{timestamp}.json"
        main_logger.info(f"Saving detailed results: {results_file}")
        print(f"Saving detailed results: {results_file}")

        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = []
            for result in all_results:
                if result:
                    result_copy = result.copy()
                    if result_copy.get('evaluation') and 'confusion_matrix' in result_copy['evaluation']:
                        result_copy['evaluation']['confusion_matrix'] = result_copy['evaluation']['confusion_matrix']
                    serializable_results.append(result_copy)
            json.dump(serializable_results, f, indent=2, default=str)

        save_confirmation = f"Detailed results JSON saved with {len(serializable_results)} experiment results"
        main_logger.info(save_confirmation)
        print(save_confirmation)

        file_info = f"\nDetailed results saved to: {results_file}\nComparison table saved to: {comparison_file}\nIndividual logs and outputs in: {base_dir}\nMain log saved to: {main_log_file}"
        main_logger.info(file_info)
        print(file_info)

        completion_msg = "\n" + "=" * 80 + "\nBENCHMARK COMPLETED SUCCESSFULLY!\n" + "=" * 80
        main_logger.info(completion_msg)
        print(completion_msg)
        
        return 0
        
    except KeyboardInterrupt:
        interrupt_msg = "\n\nBenchmark interrupted by user."
        if 'main_logger' in locals():
            main_logger.warning(interrupt_msg)
        print(interrupt_msg)
        return 1
    
    except Exception as e:
        error_msg = f"\nERROR: {e}"
        if 'main_logger' in locals():
            main_logger.error(error_msg)
            main_logger.error(traceback.format_exc())
        print(error_msg)
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
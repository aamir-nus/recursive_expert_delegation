"""
Basic Example: R.E.D. Framework Usage

This example demonstrates how to use the R.E.D. framework for text classification
with a simple synthetic dataset.
"""
import os
import sys
from pathlib import Path
import uuid

import pandas as pd
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from red.pipelines.initial_training import InitialTrainingPipeline
from red.pipelines.active_learning import ActiveLearningLoop

def create_sample_data():
    """Create sample training and unlabeled data for demonstration."""
    
    # Define sample categories and their example texts
    categories = {
        'product_review': [
            "This smartphone has excellent battery life and camera quality.",
            "The laptop is fast but the screen could be brighter.",
            "Great headphones with amazing sound quality for the price.",
            "The tablet is lightweight but lacks storage space.",
            "This camera takes stunning photos in low light conditions."
        ],
        'news_technology': [
            "Scientists develop new quantum computing breakthrough.",
            "Major tech company announces AI research partnership.",
            "New smartphone features revolutionary display technology.",
            "Cybersecurity experts warn of emerging online threats.",
            "Autonomous vehicles show promising safety test results."
        ],
        'customer_service': [
            "How can I reset my password for online banking?",
            "I need help with returning a damaged product.",
            "What are your business hours and contact information?",
            "Can you help me track my recent order shipment?",
            "I want to cancel my subscription service."
        ],
        'technical_documentation': [
            "Configure SSL certificates in Apache web server.",
            "Install Python dependencies using pip and virtual environments.",
            "Database optimization techniques for large datasets.",
            "API authentication methods and security best practices.",
            "Docker container deployment on cloud platforms."
        ],
        'academic_research': [
            "Machine learning applications in medical diagnosis.",
            "Climate change impacts on global agriculture systems.",
            "Economic effects of remote work on urban development.",
            "Psychological factors in online learning effectiveness.",
            "Renewable energy storage solutions and efficiency."
        ],
        'social_media': [
            "Just finished an amazing workout at the gym! #fitness",
            "Can't believe how beautiful the sunset is today 🌅",
            "Looking forward to the weekend with family and friends.",
            "This new coffee shop has the best latte in town!",
            "Excited to start my new job next week! #career"
        ]
    }
    
    # Create training data
    train_texts = []
    train_labels = []
    
    for category, texts in categories.items():
        for text in texts:
            train_texts.append(text)
            train_labels.append(category)
    
    # Create additional synthetic training data
    np.random.seed(42)
    for category, base_texts in categories.items():
        for _ in range(10):  # Add 10 more samples per category
            # Simple text variations
            base_text = np.random.choice(base_texts)
            variations = [
                f"In my opinion, {base_text.lower()}",
                f"I think {base_text.lower()}",
                f"It seems that {base_text.lower()}",
                f"According to experts, {base_text.lower()}",
                base_text + " This is quite interesting.",
                base_text + " What do you think?",
            ]
            train_texts.append(np.random.choice(variations))
            train_labels.append(category)
    
    # Create unlabeled data (mix of existing categories and new texts)
    unlabeled_texts = [
        "This new laptop has incredible performance for gaming.",
        "Scientists make breakthrough in renewable energy research.",
        "How do I update my account information online?",
        "Configure network security settings for enterprise systems.",
        "Educational impact of technology in remote learning environments.",
        "Excited about the new restaurant opening downtown!",
        "The phone's camera quality exceeded my expectations.",
        "Artificial intelligence advances in healthcare diagnostics.",
        "Need assistance with product warranty claim process.",
        "Software deployment strategies for microservices architecture.",
        "Environmental sustainability in urban planning research.",
        "What a fantastic concert last night! #music",
        "This tablet is perfect for digital art creation.",
        "Breakthrough in quantum encryption technology announced.",
        "Customer support was very helpful with my inquiry.",
        "Best practices for database backup and recovery.",
        "Economic analysis of global trade patterns.",
        "Beautiful weather today, perfect for outdoor activities!",
        "The smartwatch accurately tracks fitness activities.",
        "New study reveals insights into human psychology."
    ]
    
    return train_texts, train_labels, unlabeled_texts

def save_data(train_texts, train_labels, unlabeled_texts, data_dir):
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
    
    print(f"✓ Training data saved: {train_path} ({len(train_texts)} samples)")
    print(f"✓ Unlabeled data saved: {unlabeled_path} ({len(unlabeled_texts)} samples)")
    
    return str(train_path), str(unlabeled_path)

def main():
    """Run the complete R.E.D. example."""
    print("=" * 60)
    print("R.E.D. FRAMEWORK - BASIC EXAMPLE")
    print("=" * 60)

    run_id = str(uuid.uuid4())
    
    # Setup directories
    example_dir = Path(__file__).parent
    os.makedirs(example_dir / run_id, exist_ok=True)
    os.makedirs(example_dir/run_id/ "data", exist_ok=True)
    os.makedirs(example_dir/run_id/ "outputs", exist_ok=True)

    data_dir = example_dir / run_id / "data"
    output_dir = example_dir / run_id / "outputs"
    
    try:
        # Step 1: Create sample data
        print("\n1. CREATING SAMPLE DATA")
        print("-" * 30)
        
        train_texts, train_labels, unlabeled_texts = create_sample_data()
        train_path, unlabeled_path = save_data(train_texts, train_labels, unlabeled_texts, data_dir)
        
        print(f"Created {len(set(train_labels))} categories:")
        label_counts = pd.Series(train_labels).value_counts()
        for label, count in label_counts.items():
            print(f"  - {label}: {count} samples")
        
        # Step 2: Initial Training
        print("\n2. INITIAL TRAINING")
        print("-" * 30)
        
        pipeline = InitialTrainingPipeline(
            output_dir=str(output_dir)
        )
        
        initial_results = pipeline.run(
            seed_data_path=train_path,
            text_column='text',
            label_column='label'
        )
        
        if initial_results['status'] != 'success':
            print(f"✗ Initial training failed: {initial_results.get('error')}")
            return 1
        
        print(f"✓ Initial training completed:")
        print(f"  - Created {initial_results['subset_stats']['num_subsets']} subsets")
        print(f"  - Trained {initial_results['classifier_stats']['trained_classifiers']} classifiers")
        print(f"  - Time: {initial_results['total_time']:.2f} seconds")
        
        # Step 3: Active Learning
        print("\n3. ACTIVE LEARNING")
        print("-" * 30)
        
        components_dir = output_dir / "components"
        
        loop = ActiveLearningLoop(
            components_dir=str(components_dir),
            output_dir=str(output_dir)
        )
        
        al_results = loop.run(
            unlabeled_data_path=unlabeled_path,
            max_iterations=5,  # Small number for demo
            batch_size=10,
            samples_per_iteration=5
        )
        
        if al_results['status'] != 'success':
            print(f"✗ Active learning failed: {al_results.get('error')}")
            return 1
        
        print(f"✓ Active learning completed:")
        print(f"  - Iterations: {al_results['total_iterations']}")
        print(f"  - Validated samples: {al_results['total_validated_samples']}")
        print(f"  - Convergence: {'Yes' if al_results['convergence_achieved'] else 'No'}")
        print(f"  - Time: {al_results['total_time']:.2f} seconds")
        
        # Step 4: Results Summary
        print("\n4. RESULTS SUMMARY")
        print("-" * 30)
        
        if al_results['validation_history']:
            final_validation = al_results['validation_history'][-1]
            print(f"Final validation rate: {final_validation['validation_rate']:.1%}")
        
        # Show performance stats
        perf_stats = al_results.get('performance_stats', {})
        if perf_stats:
            print(f"Average iteration time: {perf_stats.get('average_iteration_time', 0):.2f}s")
            print(f"Validation efficiency: {perf_stats.get('validation_efficiency', 0):.2f} samples/sec")
        
        print(f"\nOutput files saved to: {output_dir}")
        print("- Initial training results: initial_training_summary.json")
        print("- Active learning results: active_learning_summary.json")
        print("- Final trained models: final_state/")
        print("- Final training data: final_state/final_training_data.csv")
        
        print("\n" + "=" * 60)
        print("EXAMPLE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user.")
        return 1
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())